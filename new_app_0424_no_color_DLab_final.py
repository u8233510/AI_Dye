import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split 
from recipe_recommender import build_dye_bounds, recommend_recipe

# ==========================================
# 0. 核心包裝：確保權重支援
# ==========================================
class ExplicitWeightedVotingRegressor(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)

# ==========================================
# 1. 核心數學：CMC (2:1) 色差計算
# ==========================================
def deltaE_CMC(lab_std, lab_smpl, l=2, c=1):
    L1, a1, b1 = lab_std
    L2, a2, b2 = lab_smpl
    dL, da, db = L2 - L1, a2 - a1, b2 - b1
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C2 - C1
    dH_sq = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0, dH_sq))
    SL = 0.040975 * L1 / (1 + 0.01765 * L1) if L1 >= 16 else 0.511
    SC = (0.0638 * C1 / (1 + 0.0131 * C1)) + 0.638
    H1 = np.degrees(np.arctan2(b1, a1)) % 360
    F = np.sqrt(C1**4 / (C1**4 + 1900))
    T = 0.56 + abs(0.2 * np.cos(np.radians(H1 + 168))) if 164 <= H1 <= 345 else 0.36 + abs(0.4 * np.cos(np.radians(H1 + 35)))
    SH = SC * (F * T + 1 - F)
    return np.sqrt((dL/(l*SL))**2 + (dC/(c*SC))**2 + (dH/SH)**2)

# ==========================================
# 2. 數據處理工具 (物理特性模式)
# ==========================================
def clean_dye_id(val):
    s = str(val).strip()
    if s.endswith('.0'): s = s[:-2]
    if s.lower() in ['nan', 'none', '', 'null'] or pd.isna(val): s = '無'
    return s

def detect_dye_columns(df):
    return sorted([c for c in df.columns if c.startswith('配方料號')], key=lambda x: int(x.replace('配方料號', '')))

def transform_bag_of_dyes(df_clean, dye_cols, known_dyes=None):
    df_out = df_clean.copy()
    if known_dyes is None:
        raw_values = df_out[dye_cols].values.flatten()
        known_dyes = sorted(list(set([clean_dye_id(d) for d in raw_values if clean_dye_id(d) != '無'])))
    dye_f_cols = [f'Dye_{d}' for d in known_dyes]
    for dfc in dye_f_cols: df_out[dfc] = 0.0
    for d_col in dye_cols:
        n_col = d_col.replace('料號', '濃度')
        if n_col in df_out.columns:
            for d in known_dyes:
                mask = (df_out[d_col] == d)
                df_out.loc[mask, f'Dye_{d}'] += df_out[n_col][mask]
    df_out['Total_Conc'] = df_out[dye_f_cols].sum(axis=1)
    df_out['Log_Total_Conc'] = np.log1p(df_out['Total_Conc'])
    # 物理參考特徵：標樣 LAB + DPF + OP
    base_cols = ['標準樣L', '標準樣a', '標準樣b', '色系名稱', '色系編號', 'DPF', 'OP否', 'Total_Conc', 'Log_Total_Conc']
    return df_out[base_cols + dye_f_cols], known_dyes

# ==========================================
# 3. Streamlit 介面佈局
# ==========================================
st.set_page_config(page_title="AI 專業打色系統 v20.0", layout="wide")
st.sidebar.title("⚙️ AI 系統配置")
uploaded_file = st.sidebar.file_uploader("1. 上傳打色資料", type=["csv", "xlsx"])

tab_ana, tab_feedback, tab_val, tab_recipe = st.tabs(["📊 數據分布分析", "📈 訓練回測回饋", "🔍 單筆輸入預測", "🧠 染料配方推薦"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    dye_cols = detect_dye_columns(df_raw)

    # --- 修正 1: 數據分布分析 (新增染料與色系名稱統計) ---
    with tab_ana:
        st.header("📊 全資料分布分析")
        df_ana = df_raw.copy()
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(px.histogram(df_ana, x='L', title="L* (亮/淺分布)", color_discrete_sequence=['#555555']), use_container_width=True)
        with c2: 
            df_ana['a_type'] = df_ana['a'].apply(lambda x: '紅' if x > 0 else '綠')
            st.plotly_chart(px.histogram(df_ana, x='a', color='a_type', title="a* (紅/綠分布)", color_discrete_map={'紅':'#EF553B','綠':'#00CC96'}), use_container_width=True)
        with c3:
            df_ana['b_type'] = df_ana['b'].apply(lambda x: '黃' if x > 0 else '藍')
            st.plotly_chart(px.histogram(df_ana, x='b', color='b_type', title="b* (黃/藍分布)", color_discrete_map={'黃':'#FECB52','藍':'#636EFA'}), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 染料料號對色系名稱統計表 (Dye vs. Shade Name)")
        
        # 提取所有不為「無」的染料料號
        unique_d = sorted(list(set([clean_dye_id(d) for d in df_ana[dye_cols].values.flatten() if clean_dye_id(d) != '無'])))
        shade_names = sorted(df_ana['色系名稱'].astype(str).unique())
        
        dist_data = []
        for d in unique_d:
            row = {'染料料號': d}
            for sn in shade_names:
                # 統計該染料在特定色系名稱中出現的筆數
                count = ((df_ana['色系名稱'] == sn) & (df_ana[dye_cols].apply(lambda x: d in [clean_dye_id(i) for i in x], axis=1))).sum()
                row[sn] = count
            dist_data.append(row)
        
        dist_df = pd.DataFrame(dist_data).set_index('染料料號')
        st.dataframe(dist_df.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)

        # --- 新增：染劑組合次數統計 ---
        st.markdown("---")
        st.subheader("🧪 染劑組合出現次數統計 (Dye Combinations Frequency)")
        
        # 1. 取得所有配方料號欄位名稱
        dye_id_cols = detect_dye_columns(df_ana)
        
        # 2. 定義一個函數來提取並整理每行的染劑組合
        def get_sorted_combination(row):
            # 取得該行所有染劑編號，清理掉「無」或空值
            dyes = [clean_dye_id(row[c]) for c in dye_id_cols if clean_dye_id(row[c]) != '無']
            # 排序染劑，確保組合的一致性 (例如 1191+1588 跟 1588+1191 是同一組)
            dyes.sort()
            # 用 "+" 號連接
            return "+".join(dyes) if dyes else "僅基礎藥劑(無染料)"

        # 3. 產生組合欄位並計算次數
        df_ana['染劑組合'] = df_ana.apply(get_sorted_combination, axis=1)
        comb_df = df_ana['染劑組合'].value_counts().reset_index()
        comb_df.columns = ['染劑組合', '出現次數 (筆)']
        
        # 4. 顯示統計表格，並加入百分比參考
        comb_df['佔比 (%)'] = (comb_df['出現次數 (筆)'] / len(df_ana) * 100).round(2)
        
        st.write(f"📊 總計偵測到 `{len(comb_df)}` 種不同的染劑組合。")
        st.dataframe(
            comb_df.style.background_gradient(subset=['出現次數 (筆)'], cmap='Blues'),
            use_container_width=True
        ) 


    # --- 2. 訓練核心：精準平衡加權 ---
    if st.sidebar.button("🚀 啟動模型訓練"):
        with st.spinner("AI 運算中 (平衡物理特徵學習模式)..."):
            df_train = df_raw.copy()
            # 剔除空值
            df_train = df_train.dropna(subset=['標準樣L', '標準樣a', '標準樣b', 'L', 'a', 'b', 'DPF', 'OP否', 'CMC_DE'])
            
            # 【新增修改】過濾掉 DE 過大的異常雜訊 (例如 DE > 1.5 視為非配方問題)
            df_train = df_train[df_train['CMC_DE'] <= 1.5].reset_index(drop=True)
            
            
            df_c = df_train.copy()
            for dc in dye_cols: df_c[dc] = df_c[dc].apply(clean_dye_id)
            X_bag, known_dyes = transform_bag_of_dyes(df_c, dye_cols)
            dye_bounds = build_dye_bounds(df_c, dye_cols, known_dyes, clean_dye_id)
            #Y = df_train[['L', 'a', 'b']]
            #Y = df_train[['CIE_DL', 'CIE_Da', 'CIE_Db']]
            
            # 【目標轉換】：將原本預測 a/b 軸，改為預測極坐標的 C(彩度) 與 h(色相角)
            # 1. 計算標準樣的 C 與 h
            df_train['標準樣C'] = np.sqrt(df_train['標準樣a']**2 + df_train['標準樣b']**2)
            df_train['標準樣h'] = np.degrees(np.arctan2(df_train['標準樣b'], df_train['標準樣a'])) % 360
            
            # 2. 計算實際成品的 C 與 h
            df_train['實際C'] = np.sqrt(df_train['a']**2 + df_train['b']**2)
            df_train['實際h'] = np.degrees(np.arctan2(df_train['b'], df_train['a'])) % 360
            
            # 3. 計算差距 (Delta)
            df_train['CIE_DC'] = df_train['實際C'] - df_train['標準樣C']
            
            # 4. 處理角度相減跨越 360 度的問題 (例如 359度 與 1度 其實只差 2度)
            dh = df_train['實際h'] - df_train['標準樣h']
            dh = np.where(dh > 180, dh - 360, dh)
            dh = np.where(dh < -180, dh + 360, dh)
            df_train['CIE_Dh'] = dh

            # 5. 重新定義 Y (取代原本的 Y = df_train[['CIE_DL', 'CIE_Da', 'CIE_Db']])
            Y = df_train[['CIE_DL', 'CIE_DC', 'CIE_Dh']]         
            
            
            
            X_train, X_test, Y_train, Y_test = train_test_split(X_bag, Y, test_size=0.2, random_state=42)
            
            actual_de_train = df_train.loc[X_train.index, 'CMC_DE'].values
            
            #sample_weights = np.where(actual_de_train > 0.8, 5.0, 1.0) 
            sample_weights = np.ones(len(actual_de_train))            
            
            pre = ColumnTransformer([
                #('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否']), 
                #('num', StandardScaler(), ['標準樣L', '標準樣a', '標準樣b','DPF', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes])
                
                # 【修改點】加入 '色系名稱' 到類別處理
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否', '色系名稱']), 
    
                # 【修改點】加入 '色系編號' 到數值縮放
                ('num', StandardScaler(), ['標準樣L', '標準樣a', '標準樣b', 'DPF', '色系編號', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes])
            ])
            
            #model = Pipeline([
            #    ('pre', pre), 
            #    ('reg', MultiOutputRegressor(ExplicitWeightedVotingRegressor([
            #        ('hgb', HistGradientBoostingRegressor(loss='squared_error', max_iter=1000, learning_rate=0.03, random_state=42)), 
            #        ('et', ExtraTreesRegressor(n_estimators=400, random_state=42)),
            #        ('rf', RandomForestRegressor(n_estimators=400, random_state=42))
            #    ]))) 
            #])
            
            model = Pipeline([
                ('pre', pre), 
                ('reg', MultiOutputRegressor(ExplicitWeightedVotingRegressor([
                    # 【修改點】loss 改為 'absolute_error'
                    ('hgb', HistGradientBoostingRegressor(
                       loss='absolute_error',  # 關鍵修改
                       max_iter=1000, 
                       learning_rate=0.03, 
                       random_state=42
                    )), 
                   ('et', ExtraTreesRegressor(n_estimators=400, random_state=42)),
                   ('rf', RandomForestRegressor(n_estimators=400, random_state=42))
                ]))) 
            ])
            
            model.fit(X_train, Y_train, reg__sample_weight=sample_weights) 
            preds = model.predict(X_test) 
            
            df_val_raw = df_train.loc[X_test.index].copy()
            df_fb = df_val_raw[['成品布號', 'L', 'a', 'b', 'CIE_DL', 'CIE_Da', 'CIE_Db', 'CMC_DE', '標準樣L', '標準樣a', '標準樣b']].copy().reset_index(drop=True)
            df_fb.columns = ['布號', '實際L', '實際a', '實際b', '實際DL', '實際Da', '實際Db', '實際DE', 'stdL', 'stda', 'stdb']
            #df_fb['預測L'], df_fb['預測a'], df_fb['預測b'] = preds[:,0], preds[:,1], preds[:,2]
            # 先接收預測的 Delta 值
            #df_fb['預測DL'], df_fb['預測Da'], df_fb['預測Db'] = preds[:,0], preds[:,1], preds[:,2]

            # 執行還原：預測絕對值 = 標準樣 + 預測差距
            #df_fb['預測L'] = df_fb['stdL'] + df_fb['預測DL']
            #df_fb['預測a'] = df_fb['stda'] + df_fb['預測Da']
            #df_fb['預測b'] = df_fb['stdb'] + df_fb['預測Db']
                       
            # 1. 接收模型預測出的 Delta 值 (此時模型輸出的是 DL, DC, Dh)
            df_fb['預測DL'], df_fb['預測DC'], df_fb['預測Dh'] = preds[:,0], preds[:,1], preds[:,2]
            
            # 2. 取得標準樣本身的 C (彩度) 與 h (色相角)
            std_C = np.sqrt(df_fb['stda']**2 + df_fb['stdb']**2)
            std_h = np.degrees(np.arctan2(df_fb['stdb'], df_fb['stda'])) % 360
            
            # 3. 還原出預測的絕對 C 與 h (標準樣數值 + 預測出的差距)
            pred_C = std_C + df_fb['預測DC']
            pred_C = np.maximum(pred_C, 0) # 物理限制：彩度不可能小於 0
            pred_h = (std_h + df_fb['預測Dh']) % 360
            
            # 4. 關鍵步驟：利用三角函數將 C, h 轉回 a*, b* 座標
            df_fb['預測L'] = df_fb['stdL'] + df_fb['預測DL']
            df_fb['預測a'] = pred_C * np.cos(np.radians(pred_h))
            df_fb['預測b'] = pred_C * np.sin(np.radians(pred_h))
            
            # 5. 補算預測的 Da 與 Db (為了確保與您後續的表格展示功能完全相容)
            df_fb['預測Da'] = df_fb['預測a'] - df_fb['stda']
            df_fb['預測Db'] = df_fb['預測b'] - df_fb['stdb']
            
            
            
            
            
            # 補全指標
            #df_fb['預測DL'] = df_fb['預測L'] - df_fb['stdL']
            #df_fb['預測Da'] = df_fb['預測a'] - df_fb['stda']
            #df_fb['預測Db'] = df_fb['預測b'] - df_fb['stdb']
            #df_fb['預測DE'] = [deltaE_CMC((df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']), (preds[i,0], preds[i,1], preds[i,2])) for i in range(len(df_fb))]
            df_fb['預測DE'] = [deltaE_CMC((df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']), (df_fb.loc[i, '預測L'], df_fb.loc[i, '預測a'], df_fb.loc[i, '預測b'])) for i in range(len(df_fb))]
            
            st.session_state.update({'model': model, 'kd': known_dyes, 'fb': df_fb, 'dc': dye_cols, 'df_raw': df_raw, 'dye_bounds': dye_bounds})
            st.sidebar.success(f"訓練完成！")

# --- 修正 2: 訓練回測回饋 (詳細顯示 7 項指標比對) ---
if 'fb' in st.session_state:
    with tab_feedback:
        df_fb = st.session_state['fb']
        st.header("📈 判定一致性回饋 (驗證集結果)")
        
        tp = len(df_fb[(df_fb['實際DE'] <= 0.8) & (df_fb['預測DE'] <= 0.8)])
        tn = len(df_fb[(df_fb['實際DE'] > 0.8) & (df_fb['預測DE'] > 0.8)])
        fp = len(df_fb[(df_fb['實際DE'] > 0.8) & (df_fb['預測DE'] <= 0.8)])
        fn = len(df_fb[(df_fb['實際DE'] <= 0.8) & (df_fb['預測DE'] > 0.8)])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ 雙重通過", f"{tp}")
        c2.metric("❌ 雙重失敗", f"{tn}")
        c3.metric("⚠️ 誤判通過 (危險)", f"{fp}")
        c4.metric("🔍 誤判失敗 (保守)", f"{fn}")
        
        st.info(f"### 總體判定一致性比率：{(tp + tn) / len(df_fb):.2%}")

        st.markdown("### 🔍 完整 7 項指標比對清單 (L, a, b, DL, Da, Db, DE)")
        
        # 建立展示用表格，包含實際、預測、差異
        display_df = pd.DataFrame({'布號': df_fb['布號']})
        
        metrics_map = [
            ('L', '實際L', '預測L'),
            ('a', '實際a', '預測a'),
            ('b', '實際b', '預測b'),
            ('DL', '實際DL', '預測DL'),
            ('Da', '實際Da', '預測Da'),
            ('Db', '實際Db', '預測Db'),
            ('DE', '實際DE', '預測DE')
        ]
        
        for label, act, pre in metrics_map:
            display_df[f'實際{label}'] = df_fb[act]
            display_df[f'預測{label}'] = df_fb[pre]
            display_df[f'Δ{label}差異'] = df_fb[pre] - df_fb[act]
            
        st.dataframe(display_df.round(3).style.background_gradient(subset=[f'Δ{m}差異' for m in ['L','a','b','DL','Da','Db','DE']], cmap='RdBu_r'), use_container_width=True)

        # 保留原有的圖表供深度參考
        st.markdown("---")
        st.write("#### 指標分佈圖表")
        for label, act, pre in metrics_map:
            with st.expander(f"📊 {label} 指標詳情"):
                fig = px.scatter(df_fb, x=act, y=pre, title=f"{label}：預測 vs 實際")
                fig.add_shape(type="line", x0=df_fb[act].min(), y0=df_fb[act].min(), x1=df_fb[act].max(), y1=df_fb[act].max(), line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)

# --- 分頁 3: 通用預測 ---# --- 分頁 3: 通用預測 ---
with tab_val:
    if 'model' in st.session_state:
        st.header("🔍 通用 AI 配方預測")
        col_in, col_res = st.columns([2, 1])
        
        with col_in:
            # 【關鍵修改】加入 st.form，把所有的輸入框打包起來
            with st.form("prediction_form"):
                st.write("#### 1. 物理參數 (標樣數值作為核心輸入)")
                c1, c2, c3, c4 = st.columns(4)
                v_name = c1.selectbox("色系名稱 (記錄用)", sorted(st.session_state['df_raw']['色系名稱'].astype(str).unique()))
                v_id = c2.selectbox("色系編號 (記錄用)", sorted(st.session_state['df_raw']['色系編號'].astype(str).unique()))
                v_dpf = c3.number_input("DPF", value=1.0)
                v_op = c4.selectbox("OP否", ['Y', 'N'])
                
                csL, csa, csb = st.columns(3)
                std_L_val = csL.number_input("標準樣 L*", value=50.0)
                std_a_val = csa.number_input("標準樣 a*", value=0.0)
                std_b_val = csb.number_input("標準樣 b*", value=0.0)
                
                manual_input = {}
                for i in range(1, 7):
                    cc1, cc2 = st.columns(2)
                    manual_input[f'配方料號{i}'] = cc1.selectbox(f"配方料號 {i}", ['無'] + st.session_state['kd'], key=f"p{i}")
                    manual_input[f'配方濃度{i}'] = cc2.number_input(f"配方濃度 {i}", value=0.0, format="%.4f", key=f"c{i}")

                # 【關鍵修改】按鈕必須改成 form_submit_button
                predict_btn = st.form_submit_button("🔮 執行通用預測", type="primary")

        # 只有在按下表單內的送出按鈕後，才會執行這裡的預測邏輯
        if predict_btn:
            #df_m = pd.DataFrame([{'標準樣L': std_L_val, '標準樣a': std_a_val, '標準樣b': std_b_val, 'DPF': v_dpf, 'OP否': v_op, **manual_input}])
            df_m = pd.DataFrame([{'標準樣L': std_L_val,'標準樣a': std_a_val,'標準樣b': std_b_val,'DPF': v_dpf,'OP否': v_op,'色系名稱': v_name, '色系編號': v_id, **manual_input}])
        
            
            X_m, _ = transform_bag_of_dyes(df_m, st.session_state['dc'], known_dyes=st.session_state['kd'])
           
            #m_pred = st.session_state['model'].predict(X_m)[0]
            #p_L, p_a, p_b = m_pred[0], m_pred[1], m_pred[2]
            # 接收 Delta
            #p_DL, p_Da, p_Db = m_pred[0], m_pred[1], m_pred[2]
            # 還原成座標
            #p_L = std_L_val + p_DL
            #p_a = std_a_val + p_Da
            #p_b = std_b_val + p_Db
            
            m_pred = st.session_state['model'].predict(X_m)[0]
            
            # 1. 接收預測的 Delta (DL, DC, Dh)
            p_DL, p_DC, p_Dh = m_pred[0], m_pred[1], m_pred[2]
            
            # 2. 計算標準樣目前的 C (彩度) 與 h (色相角)
            std_C = np.sqrt(std_a_val**2 + std_b_val**2)
            std_h = np.degrees(np.arctan2(std_b_val, std_a_val)) % 360
            
            # 3. 還原出預測的 C 與 h
            p_C = max(std_C + p_DC, 0) # 彩度不能小於0
            p_h = (std_h + p_Dh) % 360
            
            # 4. 利用三角函數將 C, h 轉回 a*, b* 座標
            p_L = std_L_val + p_DL
            p_a = p_C * np.cos(np.radians(p_h))
            p_b = p_C * np.sin(np.radians(p_h))
            
            
            
            
            de_val = deltaE_CMC((std_L_val, std_a_val, std_b_val), (p_L, p_a, p_b))
            
            with col_res:
                st.write("### 📊 預測結果")
                st.metric("預測 L*", f"{p_L:.2f}")
                st.metric("預測 a*", f"{p_a:.2f}")
                st.metric("預測 b*", f"{p_b:.2f}")
                st.write(f"預測 CMC DE: `{de_val:.3f}`")
                if de_val <= 0.8: 
                    st.success("✅ 合格 (DE <= 0.8)")
                else: 
                    st.error("❌ 不合格 (DE > 0.8)")
    else:
        st.warning("請先完成模型訓練。")

# --- 分頁 4: 染料配方推薦 ---
with tab_recipe:
    if 'model' in st.session_state:
        st.header("🧠 染料配方推薦")
        st.caption("輸入欄位與訓練一致，但不需輸入配方用量，由 AI 自動推薦濃度。")

        col_in, col_res = st.columns([2, 1])
        with col_in:
            with st.form("recipe_recommend_form"):
                st.write("#### 1. 物理參數 (標樣數值作為核心輸入)")
                c1, c2, c3, c4 = st.columns(4)
                r_name = c1.selectbox("色系名稱 (記錄用)", sorted(st.session_state['df_raw']['色系名稱'].astype(str).unique()), key="r_name")
                r_id = c2.selectbox("色系編號 (記錄用)", sorted(st.session_state['df_raw']['色系編號'].astype(str).unique()), key="r_id")
                r_dpf = c3.number_input("DPF", value=1.0, key="r_dpf")
                r_op = c4.selectbox("OP否", ['Y', 'N'], key="r_op")

                csL, csa, csb = st.columns(3)
                r_std_l = csL.number_input("標準樣 L*", value=50.0, key="r_std_l")
                r_std_a = csa.number_input("標準樣 a*", value=0.0, key="r_std_a")
                r_std_b = csb.number_input("標準樣 b*", value=0.0, key="r_std_b")

                st.write("#### 2. 指定最多 6 支候選染料 (不填濃度)")
                selected_dyes = []
                for i in range(1, 7):
                    dye_pick = st.selectbox(
                        f"配方料號 {i}",
                        ['無'] + st.session_state['kd'],
                        key=f"r_p{i}"
                    )
                    if dye_pick != '無' and dye_pick not in selected_dyes:
                        selected_dyes.append(dye_pick)

                n_search = st.slider("搜尋次數 (越高越精準，速度越慢)", min_value=500, max_value=5000, value=1500, step=250)
                recommend_btn = st.form_submit_button("✨ AI 推薦配方", type="primary")

        if recommend_btn:
            result = recommend_recipe(
                model=st.session_state['model'],
                transform_bag_of_dyes_fn=transform_bag_of_dyes,
                deltae_cmc_fn=deltaE_CMC,
                known_dyes=st.session_state['kd'],
                dye_cols=st.session_state['dc'],
                std_l=r_std_l,
                std_a=r_std_a,
                std_b=r_std_b,
                dpf=r_dpf,
                op=r_op,
                shade_name=r_name,
                shade_id=r_id,
                selected_dyes=selected_dyes,
                dye_bounds=st.session_state['dye_bounds'],
                n_samples=n_search,
                random_state=42,
            )

            with col_res:
                st.write("### 📦 推薦結果")
                if result is None:
                    st.warning("請至少選擇一支染料料號。")
                else:
                    rec_df = pd.DataFrame(
                        [{'染料料號': k, '推薦濃度': v} for k, v in result['recipe'].items()]
                    )
                    st.dataframe(rec_df.round(4), use_container_width=True)

                    st.write("### 📊 根據推薦配方的預測結果")
                    st.metric("L*", f"{result['L']:.2f}")
                    st.metric("a*", f"{result['a']:.2f}")
                    st.metric("b*", f"{result['b']:.2f}")
                    st.metric("DL", f"{result['DL']:.3f}")
                    st.metric("Da", f"{result['Da']:.3f}")
                    st.metric("Db", f"{result['Db']:.3f}")
                    st.metric("CMC_DE", f"{result['CMC_DE']:.3f}")

                    if result['CMC_DE'] <= 0.8:
                        st.success("✅ 推薦結果預估可合格 (DE <= 0.8)")
                    else:
                        st.error("❌ 推薦結果預估仍偏差 (DE > 0.8)")
    else:
        st.warning("請先完成模型訓練。")
