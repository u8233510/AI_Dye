import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import clean_dye_id, detect_dye_columns


@st.cache_data(show_spinner=False)
def _build_dye_vs_shade_table(df_ana, dye_cols):
    shade_series = df_ana['色系名稱'].astype(str)
    shade_names = sorted(shade_series.unique())

    long_df = (
        df_ana[list(dye_cols) + ['色系名稱']]
        .melt(id_vars='色系名稱', value_vars=dye_cols, value_name='dye')
        .assign(
            dye=lambda d: d['dye'].map(clean_dye_id),
            色系名稱=lambda d: d['色系名稱'].astype(str),
        )
    )
    long_df = long_df[long_df['dye'] != '無']

    if long_df.empty:
        return pd.DataFrame(columns=shade_names)

    dist_df = pd.crosstab(long_df['dye'], long_df['色系名稱'])
    dist_df.index.name = '染料料號'
    dist_df = dist_df.reindex(columns=shade_names, fill_value=0)
    return dist_df.sort_index()


@st.cache_data(show_spinner=False)
def _build_combination_stats(df_ana, dye_id_cols):
    normalized = df_ana[dye_id_cols].applymap(clean_dye_id)

    def _row_combination(row):
        dyes = sorted([d for d in row if d != '無'])
        return '+'.join(dyes) if dyes else '僅基礎藥劑(無染料)'

    comb_series = normalized.apply(_row_combination, axis=1)
    comb_df = comb_series.value_counts().rename_axis('染劑組合').reset_index(name='出現次數 (筆)')
    comb_df['佔比 (%)'] = (comb_df['出現次數 (筆)'] / len(df_ana) * 100).round(2)
    return comb_df


def render_data_distribution(tab_ana, df_raw, dye_cols):
    with tab_ana:
        st.header('📊 全資料分布分析')
        df_ana = df_raw.copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                px.histogram(df_ana, x='L', title='L* (亮/淺分布)', color_discrete_sequence=['#555555']),
                use_container_width=True,
            )
        with c2:
            df_ana['a_type'] = df_ana['a'].apply(lambda x: '紅' if x > 0 else '綠')
            st.plotly_chart(
                px.histogram(df_ana, x='a', color='a_type', title='a* (紅/綠分布)', color_discrete_map={'紅': '#EF553B', '綠': '#00CC96'}),
                use_container_width=True,
            )
        with c3:
            df_ana['b_type'] = df_ana['b'].apply(lambda x: '黃' if x > 0 else '藍')
            st.plotly_chart(
                px.histogram(df_ana, x='b', color='b_type', title='b* (黃/藍分布)', color_discrete_map={'黃': '#FECB52', '藍': '#636EFA'}),
                use_container_width=True,
            )

        st.markdown('---')
        st.subheader('📋 染料料號對色系名稱統計表 (Dye vs. Shade Name)')
        dist_df = _build_dye_vs_shade_table(df_ana, tuple(dye_cols))
        st.dataframe(dist_df.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)

        st.markdown('---')
        st.subheader('🧪 染劑組合出現次數統計 (Dye Combinations Frequency)')

        dye_id_cols = detect_dye_columns(df_ana)
        comb_df = _build_combination_stats(df_ana, tuple(dye_id_cols))

        st.write(f'📊 總計偵測到 `{len(comb_df)}` 種不同的染劑組合。')
        st.dataframe(
            comb_df.style.background_gradient(subset=['出現次數 (筆)'], cmap='Blues'),
            use_container_width=True,
        )
