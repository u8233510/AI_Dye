import streamlit as st

from display_module import render_data_distribution
from input_module import load_uploaded_data, render_input_sidebar
from model_utils import detect_dye_columns
from prediction_module import render_prediction
from recommendation_module import render_recommendation
from training_module import render_training_button
from training_results_module import render_training_feedback


st.set_page_config(page_title="AI 專業打色系統 v20.0", layout="wide")

uploaded_file = render_input_sidebar()
tab_ana, tab_feedback, tab_val, tab_rec = st.tabs(["📊 數據分布分析", "📈 訓練回測回饋", "🔍 單筆輸入預測", "🧪 配方推薦"])

if uploaded_file:
    df_raw = load_uploaded_data(uploaded_file)
    dye_cols = detect_dye_columns(df_raw)

    render_data_distribution(tab_ana, df_raw, dye_cols)
    render_training_button(df_raw, dye_cols)

if 'fb' in st.session_state:
    render_training_feedback(tab_feedback, st.session_state['fb'])

render_prediction(tab_val)
render_recommendation(tab_rec)
