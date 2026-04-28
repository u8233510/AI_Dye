import pandas as pd
import streamlit as st

from model_utils import deltaE_CMC, transform_bag_of_dyes


def render_prediction(tab_val):
    with tab_val:
        if 'model' not in st.session_state:
            st.warning("請先完成模型訓練。")
            return

        st.header("🔍 通用 AI 配方預測")
        col_in, col_res = st.columns([2, 1])

        with col_in:
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

                predict_btn = st.form_submit_button("🔮 執行通用預測", type="primary")

        if predict_btn:
            df_m = pd.DataFrame(
                [
                    {
                        '標準樣L': std_L_val,
                        '標準樣a': std_a_val,
                        '標準樣b': std_b_val,
                        'DPF': v_dpf,
                        'OP否': v_op,
                        '色系名稱': v_name,
                        '色系編號': v_id,
                        **manual_input,
                    }
                ]
            )

            X_m, _ = transform_bag_of_dyes(df_m, st.session_state['dc'], known_dyes=st.session_state['kd'])
            m_pred = st.session_state['model'].predict(X_m)[0]
            p_DL, p_Da, p_Db = m_pred[0], m_pred[1], m_pred[2]
            p_L = std_L_val + p_DL
            p_a = std_a_val + p_Da
            p_b = std_b_val + p_Db
            de_from_lab = deltaE_CMC((std_L_val, std_a_val, std_b_val), (p_L, p_a, p_b))
            if 'de_model' in st.session_state:
                de_raw = max(float(st.session_state['de_model'].predict(X_m)[0]), 0.0)
                if 'de_calibration' in st.session_state:
                    cal = st.session_state['de_calibration']
                    de_global = max(float(de_raw * float(cal.get('slope', 1.0)) + float(cal.get('intercept', 0.0))), 0.0)
                else:
                    de_global = de_raw

                if 'local_de_model' in st.session_state:
                    local_pack = st.session_state['local_de_model']
                    X_local = pd.get_dummies(X_m, columns=local_pack['cat_cols'], dummy_na=False)
                    X_local = X_local.reindex(columns=local_pack['feature_columns'], fill_value=0.0)
                    de_local = max(float(local_pack['model'].predict(X_local)[0]), 0.0)
                    de_val = max(float(0.45 * de_global + 0.55 * de_local), 0.0)
                    de_source = '模型直預測(全域+近鄰混合)'
                else:
                    de_val = de_global
                    de_source = '模型直預測(全域校正)'
            else:
                de_val = de_from_lab
                de_source = '由 Lab 換算'

            with col_res:
                st.write("### 📊 預測結果")
                st.metric("預測 L*", f"{p_L:.2f}")
                st.metric("預測 a*", f"{p_a:.2f}")
                st.metric("預測 b*", f"{p_b:.2f}")
                st.write(f"預測 CMC DE ({de_source}): `{de_val:.3f}`")
                if de_val <= 0.8:
                    st.success("✅ 合格 (DE <= 0.8)")
                else:
                    st.error("❌ 不合格 (DE > 0.8)")
