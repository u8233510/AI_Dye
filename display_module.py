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
    selected_cols = [col for col in dye_id_cols if col in df_ana.columns]
    if not selected_cols:
        return pd.DataFrame(
            [{'染劑組合': '僅基礎藥劑(無染料)', '出現次數 (筆)': len(df_ana), '佔比 (%)': 100.0}]
        )

    normalized = df_ana[selected_cols].applymap(clean_dye_id)

    def _row_combination(row):
        dyes = sorted([d for d in row if d != '無'])
        return '+'.join(dyes) if dyes else '僅基礎藥劑(無染料)'

    comb_series = normalized.apply(_row_combination, axis=1)
    comb_df = comb_series.value_counts().rename_axis('染劑組合').reset_index(name='出現次數 (筆)')
    comb_df['佔比 (%)'] = (comb_df['出現次數 (筆)'] / len(df_ana) * 100).round(2)
    return comb_df



def _render_delta_distribution(df_ana):
    delta_specs = [
        ('DL', ['DL', 'CIE_DL'], 'DL 分布', '#636EFA'),
        ('Da', ['Da', 'CIE_Da'], 'Da 分布', '#EF553B'),
        ('Db', ['Db', 'CIE_Db'], 'Db 分布', '#00CC96'),
        ('DE', ['DE', 'CMC_DE'], 'DE 分布', '#AB63FA'),
    ]

    available = []
    for label, candidates, title, color in delta_specs:
        source_col = next((col for col in candidates if col in df_ana.columns), None)
        if source_col:
            available.append((label, source_col, title, color))

    if not available:
        st.info('目前資料中沒有 DL/Da/Db/DE（或 CIE_DL/CIE_Da/CIE_Db/CMC_DE）欄位，略過該分布圖。')
        return

    st.markdown('---')
    st.subheader('🎯 訓練資料誤差指標分布 (DL / Da / Db / DE)')

    st.caption('可用滑鼠框選區域縮放，雙擊可重置；下方每個指標皆為展開大圖。')

    tabs = st.tabs([f'{label} 分布' for label, _, _, _ in available])
    for tab, (label, source_col, title, color) in zip(tabs, available):
        with tab:
            title_with_source = f"{title}（來源欄位：{source_col}）"
            fig = px.histogram(df_ana, x=source_col, title=title_with_source, color_discrete_sequence=[color])
            fig.update_layout(
                bargap=0.05,
                height=520,
                xaxis=dict(rangeslider=dict(visible=True)),
                dragmode='zoom',
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_training_feature_columns(df_ana, dye_cols):
    st.markdown('---')
    st.subheader('🧾 目前訓練模型輸入欄位')

    base_features = [
        'OP否',
        '色系名稱',
        '標準樣L',
        '標準樣a',
        '標準樣b',
        'DPF',
        '色系編號',
        'Total_Conc',
        'Log_Total_Conc',
    ]
    available_base = [col for col in base_features if col in df_ana.columns]

    dye_feature_count = len([col for col in dye_cols if col in df_ana.columns])

    st.write('固定欄位（含類別 One-Hot 後輸入模型）:')
    st.code(', '.join(available_base) if available_base else '（無可用固定欄位）')

    if dye_feature_count > 0:
        st.write(f'染料欄位（Bag-of-Dyes 轉換）: 偵測到 {dye_feature_count} 欄，會展開為 `Dye_<料號>` 特徵。')
    else:
        st.write('染料欄位（Bag-of-Dyes 轉換）: 未偵測到可用染料欄位。')

    st.caption('目標欄位（Y）為：CIE_DL、CIE_Da、CIE_Db；CMC_DE 另訓練 DE 專屬模型。')


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

        _render_delta_distribution(df_ana)
        _render_training_feature_columns(df_ana, dye_cols)

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
