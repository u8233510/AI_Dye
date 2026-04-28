import pandas as pd
import plotly.express as px
import streamlit as st


def _calc_fixed_report(df_fb, de_col):
    mae_dl = float((df_fb['預測DL'] - df_fb['實際DL']).abs().mean())
    mae_da = float((df_fb['預測Da'] - df_fb['實際Da']).abs().mean())
    mae_db = float((df_fb['預測Db'] - df_fb['實際Db']).abs().mean())
    de_mae = float((df_fb[de_col] - df_fb['實際DE']).abs().mean())

    actual_over = df_fb['實際DE'] > 0.8
    pred_over = df_fb[de_col] > 0.8
    miss_mask = actual_over & (~pred_over)
    miss_rate = float(miss_mask.sum() / actual_over.sum()) if actual_over.sum() > 0 else 0.0

    high_risk_cut = float(df_fb['實際DE'].quantile(0.9))
    high_risk_mask = df_fb['實際DE'] >= high_risk_cut
    high_risk_mae = float((df_fb.loc[high_risk_mask, de_col] - df_fb.loc[high_risk_mask, '實際DE']).abs().mean()) if high_risk_mask.any() else 0.0

    return {
        'MAE_DL': mae_dl,
        'MAE_Da': mae_da,
        'MAE_Db': mae_db,
        'DE_MAE': de_mae,
        'DE漏判率_>0.8': miss_rate,
        '高風險區MAE_前10%DE': high_risk_mae,
        '高風險區門檻DE': high_risk_cut,
    }


def render_training_feedback(tab_feedback, df_fb):
    with tab_feedback:
        st.header("📈 判定一致性回饋 (驗證集結果)")

        de_col = '預測DE_模型' if '預測DE_模型' in df_fb.columns else '預測DE'
        tp = len(df_fb[(df_fb['實際DE'] <= 0.8) & (df_fb[de_col] <= 0.8)])
        tn = len(df_fb[(df_fb['實際DE'] > 0.8) & (df_fb[de_col] > 0.8)])
        fp = len(df_fb[(df_fb['實際DE'] > 0.8) & (df_fb[de_col] <= 0.8)])
        fn = len(df_fb[(df_fb['實際DE'] <= 0.8) & (df_fb[de_col] > 0.8)])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ 雙重通過", f"{tp}")
        c2.metric("❌ 雙重失敗", f"{tn}")
        c3.metric("⚠️ 誤判通過 (危險)", f"{fp}")
        c4.metric("🔍 誤判失敗 (保守)", f"{fn}")

        st.info(f"### 總體判定一致性比率：{(tp + tn) / len(df_fb):.2%}")
        report = _calc_fixed_report(df_fb, de_col)

        st.markdown("### 📌 固定評估報表")
        r1, r2, r3 = st.columns(3)
        r1.metric("MAE_DL", f"{report['MAE_DL']:.4f}")
        r2.metric("MAE_Da", f"{report['MAE_Da']:.4f}")
        r3.metric("MAE_Db", f"{report['MAE_Db']:.4f}")
        r4, r5, r6 = st.columns(3)
        r4.metric("DE_MAE", f"{report['DE_MAE']:.4f}")
        r5.metric("DE>0.8 漏判率", f"{report['DE漏判率_>0.8']:.2%}")
        r6.metric("高風險區 MAE (前10% DE)", f"{report['高風險區MAE_前10%DE']:.4f}")
        st.caption(f"高風險區定義：實際 DE >= P90（門檻 {report['高風險區門檻DE']:.4f}）")

        st.markdown("### 🔍 完整 7 項指標比對清單 (L, a, b, DL, Da, Db, DE)")
        display_df = pd.DataFrame({'布號': df_fb['布號']})

        metrics_map = [
            ('L', '實際L', '預測L'),
            ('a', '實際a', '預測a'),
            ('b', '實際b', '預測b'),
            ('DL', '實際DL', '預測DL'),
            ('Da', '實際Da', '預測Da'),
            ('Db', '實際Db', '預測Db'),
            ('DE', '實際DE', de_col),
        ]

        for label, act, pre in metrics_map:
            display_df[f'實際{label}'] = df_fb[act]
            display_df[f'預測{label}'] = df_fb[pre]
            display_df[f'Δ{label}差異'] = df_fb[pre] - df_fb[act]

        st.dataframe(
            display_df.round(3).style.background_gradient(
                subset=[f'Δ{m}差異' for m in ['L', 'a', 'b', 'DL', 'Da', 'Db', 'DE']],
                cmap='RdBu_r',
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.write("#### 指標分佈圖表")
        for label, act, pre in metrics_map:
            with st.expander(f"📊 {label} 指標詳情"):
                fig = px.scatter(df_fb, x=act, y=pre, title=f"{label}：預測 vs 實際")
                fig.add_shape(
                    type="line",
                    x0=df_fb[act].min(),
                    y0=df_fb[act].min(),
                    x1=df_fb[act].max(),
                    y1=df_fb[act].max(),
                    line=dict(color="Red", dash="dash"),
                )
                st.plotly_chart(fig, use_container_width=True)
