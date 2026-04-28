import pandas as pd
import plotly.express as px
import streamlit as st


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
