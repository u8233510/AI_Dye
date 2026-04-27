import numpy as np
import pandas as pd
import streamlit as st

from model_utils import deltaE_CMC, transform_bag_of_dyes


def _collect_concentration_stats(df_raw, dye_cols):
    conc_stats = {}
    for d_col in dye_cols:
        c_col = d_col.replace('料號', '濃度')
        if c_col not in df_raw.columns:
            continue
        for dye in df_raw[d_col].astype(str).str.strip().values:
            if dye in {'', 'nan', 'None', '無'}:
                continue
            mask = df_raw[d_col].astype(str).str.strip() == dye
            vals = pd.to_numeric(df_raw.loc[mask, c_col], errors='coerce').dropna()
            vals = vals[vals > 0]
            if vals.empty:
                continue
            if dye not in conc_stats:
                conc_stats[dye] = []
            conc_stats[dye].extend(vals.tolist())

    summary = {}
    for dye, vals in conc_stats.items():
        arr = np.asarray(vals, dtype=float)
        summary[dye] = {
            'q95': float(np.quantile(arr, 0.95)),
            'max': float(arr.max()),
        }
    return summary


def _candidate_upper_bounds(selected_dyes, conc_summary):
    upper_bounds = []
    for dye in selected_dyes:
        if dye == '無':
            upper_bounds.append(0.0)
            continue
        if dye in conc_summary:
            bound = max(conc_summary[dye]['q95'] * 1.2, conc_summary[dye]['max'] * 0.8)
        else:
            bound = 5.0
        upper_bounds.append(float(np.clip(bound, 0.05, 20.0)))
    return np.array(upper_bounds, dtype=float)


def _predict_with_recipe(base_inputs, recipe_concs, selected_dyes):
    manual_input = {}
    for i in range(1, 7):
        manual_input[f'配方料號{i}'] = selected_dyes[i - 1]
        manual_input[f'配方濃度{i}'] = float(recipe_concs[i - 1])

    df_m = pd.DataFrame([
        {
            '標準樣L': base_inputs['標準樣L'],
            '標準樣a': base_inputs['標準樣a'],
            '標準樣b': base_inputs['標準樣b'],
            'DPF': base_inputs['DPF'],
            'OP否': base_inputs['OP否'],
            '色系名稱': base_inputs['色系名稱'],
            '色系編號': base_inputs['色系編號'],
            **manual_input,
        }
    ])

    X_m, _ = transform_bag_of_dyes(df_m, st.session_state['dc'], known_dyes=st.session_state['kd'])
    p_DL, p_Da, p_Db = st.session_state['model'].predict(X_m)[0]

    p_L = base_inputs['標準樣L'] + p_DL
    p_a = base_inputs['標準樣a'] + p_Da
    p_b = base_inputs['標準樣b'] + p_Db
    de_val = deltaE_CMC((base_inputs['標準樣L'], base_inputs['標準樣a'], base_inputs['標準樣b']), (p_L, p_a, p_b))

    return {
        'pred_L': float(p_L),
        'pred_a': float(p_a),
        'pred_b': float(p_b),
        'pred_DL': float(p_DL),
        'pred_Da': float(p_Da),
        'pred_Db': float(p_Db),
        'pred_DE': float(de_val),
    }


def _recommend_recipe(base_inputs, selected_dyes):
    conc_summary = _collect_concentration_stats(st.session_state['df_raw'], st.session_state['dc'])
    upper_bounds = _candidate_upper_bounds(selected_dyes, conc_summary)

    rng = np.random.default_rng(42)
    n_slots = len(selected_dyes)

    best_x = np.zeros(n_slots, dtype=float)
    best_result = _predict_with_recipe(base_inputs, best_x, selected_dyes)
    best_score = best_result['pred_DE'] + 0.01 * best_x.sum()

    n_samples = 1400
    random_candidates = rng.random((n_samples, n_slots)) * upper_bounds
    random_candidates = np.vstack([np.zeros((1, n_slots)), random_candidates])

    for cand in random_candidates:
        cand = np.where(np.array(selected_dyes) == '無', 0.0, cand)
        result = _predict_with_recipe(base_inputs, cand, selected_dyes)
        score = result['pred_DE'] + 0.01 * cand.sum()
        if score < best_score:
            best_score = score
            best_x = cand.copy()
            best_result = result

    for _ in range(8):
        noise = rng.normal(loc=0.0, scale=0.1, size=(300, n_slots))
        local_candidates = np.clip(best_x * (1 + noise), 0.0, upper_bounds)
        for cand in local_candidates:
            cand = np.where(np.array(selected_dyes) == '無', 0.0, cand)
            result = _predict_with_recipe(base_inputs, cand, selected_dyes)
            score = result['pred_DE'] + 0.01 * cand.sum()
            if score < best_score:
                best_score = score
                best_x = cand.copy()
                best_result = result

    return best_x, best_result


def render_recommendation(tab_recommend):
    with tab_recommend:
        if 'model' not in st.session_state:
            st.warning('請先完成模型訓練。')
            return

        st.header('🧪 AI 染料配方推薦')
        st.caption('輸入欄位與訓練資料一致，但不需要填寫配方濃度，系統會自動推薦。')

        col_in, col_res = st.columns([2, 1])

        with col_in:
            with st.form('recommendation_form'):
                st.write('#### 1. 物理參數 (標樣數值作為核心輸入)')
                c1, c2, c3, c4 = st.columns(4)
                v_name = c1.selectbox('色系名稱 (記錄用)', sorted(st.session_state['df_raw']['色系名稱'].astype(str).unique()), key='r_name')
                v_id = c2.selectbox('色系編號 (記錄用)', sorted(st.session_state['df_raw']['色系編號'].astype(str).unique()), key='r_id')
                v_dpf = c3.number_input('DPF', value=1.0, key='r_dpf')
                v_op = c4.selectbox('OP否', ['Y', 'N'], key='r_op')

                csL, csa, csb = st.columns(3)
                std_L_val = csL.number_input('標準樣 L*', value=50.0, key='r_std_l')
                std_a_val = csa.number_input('標準樣 a*', value=0.0, key='r_std_a')
                std_b_val = csb.number_input('標準樣 b*', value=0.0, key='r_std_b')

                st.write('#### 2. 配方料號 (由 AI 推薦濃度)')
                selected_dyes = []
                for i in range(1, 7):
                    selected = st.selectbox(
                        f'配方料號 {i}',
                        ['無'] + st.session_state['kd'],
                        key=f'rp{i}',
                    )
                    selected_dyes.append(selected)

                recommend_btn = st.form_submit_button('🤖 產生推薦配方', type='primary')

        if recommend_btn:
            base_inputs = {
                '標準樣L': std_L_val,
                '標準樣a': std_a_val,
                '標準樣b': std_b_val,
                'DPF': v_dpf,
                'OP否': v_op,
                '色系名稱': v_name,
                '色系編號': v_id,
            }

            rec_conc, pred = _recommend_recipe(base_inputs, selected_dyes)

            with col_res:
                st.write('### 📊 推薦後預測結果')
                st.metric('預測 L*', f"{pred['pred_L']:.2f}")
                st.metric('預測 a*', f"{pred['pred_a']:.2f}")
                st.metric('預測 b*', f"{pred['pred_b']:.2f}")
                st.metric('預測 DL', f"{pred['pred_DL']:.3f}")
                st.metric('預測 Da', f"{pred['pred_Da']:.3f}")
                st.metric('預測 Db', f"{pred['pred_Db']:.3f}")
                st.write(f"預測 CMC_DE: `{pred['pred_DE']:.3f}`")
                if pred['pred_DE'] <= 0.8:
                    st.success('✅ 合格 (DE <= 0.8)')
                else:
                    st.error('❌ 不合格 (DE > 0.8)')

            recipe_df = pd.DataFrame(
                {
                    '配方料號': selected_dyes,
                    'AI 推薦濃度': [round(float(v), 4) for v in rec_conc],
                }
            )
            st.markdown('#### 🧾 AI 推薦配方')
            st.dataframe(recipe_df, use_container_width=True)
