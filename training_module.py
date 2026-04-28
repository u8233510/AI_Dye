import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import clean_dye_id, deltaE_CMC, transform_bag_of_dyes

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ARTIFACT_DIR = Path('.model_cache')


class ExplicitWeightedVotingRegressor(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)


MODEL_LABELS = {
    'current_ensemble': '目前模型 (Voting Ensemble)',
    'automl_lite': 'AutoML-lite (RandomizedSearchCV)',
    'catboost': 'CatBoost (Gradient Boosting)',
}


def _build_preprocessor(known_dyes):
    return ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否', '色系名稱']),
            (
                'num',
                StandardScaler(),
                ['標準樣L', '標準樣a', '標準樣b', 'DPF', '色系編號', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes],
            ),
        ]
    )


def _build_current_model(pre, model_params):
    hgb = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=int(model_params['hgb_max_iter']),
        learning_rate=float(model_params['hgb_learning_rate']),
        random_state=42,
    )
    et = ExtraTreesRegressor(
        n_estimators=int(model_params['et_n_estimators']),
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=int(model_params['rf_n_estimators']),
        random_state=42,
    )

    return Pipeline(
        [
            ('pre', pre),
            ('reg', MultiOutputRegressor(ExplicitWeightedVotingRegressor([('hgb', hgb), ('et', et), ('rf', rf)]))),
        ]
    )


def _build_automl_pipeline(pre):
    return Pipeline(
        [
            ('pre', pre),
            ('reg', MultiOutputRegressor(RandomForestRegressor(random_state=42))),
        ]
    )


def _build_catboost_pipeline(pre, model_params):
    if CatBoostRegressor is None:
        raise ImportError('CatBoost 未安裝，請先安裝 catboost 套件。')

    cat = CatBoostRegressor(
        loss_function='MAE',
        iterations=int(model_params.get('cat_iterations', 1200)),
        learning_rate=float(model_params.get('cat_learning_rate', 0.03)),
        depth=int(model_params.get('cat_depth', 8)),
        l2_leaf_reg=float(model_params.get('cat_l2_leaf_reg', 3.0)),
        random_seed=42,
        verbose=False,
    )

    return Pipeline(
        [
            ('pre', pre),
            ('reg', MultiOutputRegressor(cat)),
        ]
    )


def _fit_automl_lite(model, X_train, Y_train, sample_weights, model_params):
    search_space = [
        {
            'reg__estimator': [RandomForestRegressor(random_state=42)],
            'reg__estimator__n_estimators': [200, 300, 400, 500, 700],
            'reg__estimator__max_depth': [None, 8, 12, 20],
            'reg__estimator__min_samples_split': [2, 4, 8],
        },
        {
            'reg__estimator': [ExtraTreesRegressor(random_state=42)],
            'reg__estimator__n_estimators': [200, 300, 400, 500, 700],
            'reg__estimator__max_depth': [None, 8, 12, 20],
            'reg__estimator__min_samples_split': [2, 4, 8],
        },
        {
            'reg__estimator': [HistGradientBoostingRegressor(loss='absolute_error', random_state=42)],
            'reg__estimator__max_iter': [300, 500, 800, 1000],
            'reg__estimator__learning_rate': [0.01, 0.03, 0.05, 0.08],
            'reg__estimator__max_depth': [None, 4, 8],
        },
    ]

    automl = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=int(model_params['n_iter']),
        cv=int(model_params['cv_folds']),
        scoring=model_params['scoring'],
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    automl.fit(X_train, Y_train, reg__sample_weight=sample_weights)
    return automl.best_estimator_, automl.best_params_, automl.best_score_


def _dataset_fingerprint(df_raw, dye_cols):
    safe = df_raw.copy().fillna('')
    table_hash = hashlib.sha256(pd.util.hash_pandas_object(safe, index=True).values.tobytes()).hexdigest()
    payload = {'table_hash': table_hash, 'dye_cols': list(dye_cols), 'shape': list(df_raw.shape)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _training_signature(df_raw, dye_cols, model_type, model_params):
    payload = {
        'dataset': _dataset_fingerprint(df_raw, dye_cols),
        'model_type': model_type,
        'model_params': model_params,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _load_cached_state(signature):
    artifact_path = ARTIFACT_DIR / f'{signature}.joblib'
    if artifact_path.exists():
        return joblib.load(artifact_path)
    return None


def _save_cached_state(signature, state):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f'{signature}.joblib'
    joblib.dump(state, artifact_path)


def train_model(df_raw, dye_cols, model_type='current_ensemble', model_params=None):
    model_params = model_params or {}

    df_train = df_raw.copy().dropna(
        subset=['標準樣L', '標準樣a', '標準樣b', 'L', 'a', 'b', 'DPF', 'OP否', 'CMC_DE']
    )

    df_c = df_train.copy()
    for dc in dye_cols:
        df_c[dc] = df_c[dc].apply(clean_dye_id)

    X_bag, known_dyes = transform_bag_of_dyes(df_c, dye_cols)
    Y = df_train[['CIE_DL', 'CIE_Da', 'CIE_Db']]

    X_train, X_test, Y_train, Y_test = train_test_split(X_bag, Y, test_size=0.2, random_state=42)
    actual_de_train = df_train.loc[X_train.index, 'CMC_DE'].values
    sample_weights = np.where(actual_de_train > float(model_params.get('high_de_threshold', 0.8)), float(model_params.get('high_de_weight', 5.0)), 1.0)

    pre = _build_preprocessor(known_dyes)

    automl_info = None
    if model_type == 'automl_lite':
        base_pipeline = _build_automl_pipeline(pre)
        model, best_params, best_score = _fit_automl_lite(base_pipeline, X_train, Y_train, sample_weights, model_params)
        automl_info = {
            'best_params': best_params,
            'best_cv_score': best_score,
        }
    elif model_type == 'catboost':
        model = _build_catboost_pipeline(pre, model_params)
        model.fit(X_train, Y_train, reg__sample_weight=sample_weights)
    else:
        model = _build_current_model(pre, model_params)
        model.fit(X_train, Y_train, reg__sample_weight=sample_weights)

    # 額外訓練一個 CMC_DE 直接回歸模型，避免由 DL/Da/Db 間接換算時放大誤差
    pre_de = _build_preprocessor(known_dyes)
    if model_type == 'catboost':
        de_model = Pipeline(
            [
                ('pre', pre_de),
                (
                    'reg',
                    CatBoostRegressor(
                        loss_function='MAE',
                        iterations=int(model_params.get('cat_iterations', 1200)),
                        learning_rate=float(model_params.get('cat_learning_rate', 0.03)),
                        depth=int(model_params.get('cat_depth', 8)),
                        l2_leaf_reg=float(model_params.get('cat_l2_leaf_reg', 3.0)),
                        random_seed=42,
                        verbose=False,
                    ),
                ),
            ]
        )
    else:
        de_model = Pipeline(
            [
                ('pre', pre_de),
                ('reg', HistGradientBoostingRegressor(loss='absolute_error', max_iter=700, learning_rate=0.03, random_state=42)),
            ]
        )
    de_model.fit(X_train, df_train.loc[X_train.index, 'CMC_DE'], reg__sample_weight=sample_weights)

    preds = model.predict(X_test)

    df_val_raw = df_train.loc[X_test.index].copy()
    df_fb = df_val_raw[
        ['成品布號', 'L', 'a', 'b', 'CIE_DL', 'CIE_Da', 'CIE_Db', 'CMC_DE', '標準樣L', '標準樣a', '標準樣b']
    ].copy().reset_index(drop=True)
    df_fb.columns = ['布號', '實際L', '實際a', '實際b', '實際DL', '實際Da', '實際Db', '實際DE', 'stdL', 'stda', 'stdb']

    df_fb['預測DL'], df_fb['預測Da'], df_fb['預測Db'] = preds[:, 0], preds[:, 1], preds[:, 2]
    df_fb['預測L'] = df_fb['stdL'] + df_fb['預測DL']
    df_fb['預測a'] = df_fb['stda'] + df_fb['預測Da']
    df_fb['預測b'] = df_fb['stdb'] + df_fb['預測Db']
    df_fb['預測DE'] = [
        deltaE_CMC(
            (df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']),
            (df_fb.loc[i, '預測L'], df_fb.loc[i, '預測a'], df_fb.loc[i, '預測b']),
        )
        for i in range(len(df_fb))
    ]
    df_fb['預測DE_模型'] = np.clip(de_model.predict(X_test), 0, None)

    return {
        'model': model,
        'de_model': de_model,
        'kd': known_dyes,
        'fb': df_fb,
        'dc': dye_cols,
        'df_raw': df_raw,
        'model_type': model_type,
        'model_label': MODEL_LABELS.get(model_type, model_type),
        'model_params': model_params,
        'automl_info': automl_info,
    }


def _render_model_params(model_type):
    params = {}

    st.sidebar.markdown('### 🧩 模型參數設定')

    params['high_de_threshold'] = st.sidebar.number_input('高誤差閾值 (CMC_DE)', min_value=0.0, max_value=10.0, value=0.8, step=0.1)
    params['high_de_weight'] = st.sidebar.number_input('高誤差樣本權重', min_value=1.0, max_value=20.0, value=5.0, step=0.5)

    if model_type == 'current_ensemble':
        st.sidebar.caption('目前模型：Voting Ensemble')
        params['hgb_max_iter'] = st.sidebar.slider('HGB max_iter', min_value=200, max_value=2000, value=1000, step=100)
        params['hgb_learning_rate'] = st.sidebar.number_input('HGB learning_rate', min_value=0.001, max_value=0.3, value=0.03, step=0.005, format='%.3f')
        params['et_n_estimators'] = st.sidebar.slider('ExtraTrees n_estimators', min_value=100, max_value=1000, value=400, step=50)
        params['rf_n_estimators'] = st.sidebar.slider('RandomForest n_estimators', min_value=100, max_value=1000, value=400, step=50)
    elif model_type == 'automl_lite':
        st.sidebar.caption('AutoML-lite：對多個模型做隨機搜尋')
        params['n_iter'] = st.sidebar.slider('搜尋次數 (n_iter)', min_value=5, max_value=60, value=20, step=5)
        params['cv_folds'] = st.sidebar.selectbox('交叉驗證折數', [3, 4, 5], index=0)
        params['scoring'] = st.sidebar.selectbox('評分指標', ['neg_mean_absolute_error', 'neg_root_mean_squared_error'], index=0)
    elif model_type == 'catboost':
        st.sidebar.caption('CatBoost：先做穩定版，再與其他模型比較')
        params['cat_iterations'] = st.sidebar.slider('CatBoost iterations', min_value=300, max_value=3000, value=1200, step=100)
        params['cat_learning_rate'] = st.sidebar.number_input(
            'CatBoost learning_rate',
            min_value=0.005,
            max_value=0.3,
            value=0.03,
            step=0.005,
            format='%.3f',
        )
        params['cat_depth'] = st.sidebar.slider('CatBoost depth', min_value=4, max_value=12, value=8, step=1)
        params['cat_l2_leaf_reg'] = st.sidebar.number_input('CatBoost l2_leaf_reg', min_value=1.0, max_value=20.0, value=3.0, step=0.5)

    return params


def render_training_button(df_raw, dye_cols):
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 🤖 訓練模型選擇')
    if CatBoostRegressor is None:
        st.sidebar.warning('未偵測到 catboost 套件；若選 CatBoost 將無法訓練。')

    model_type = st.sidebar.selectbox('模型類型', options=list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    model_params = _render_model_params(model_type)
    signature = _training_signature(df_raw, dye_cols, model_type, model_params)

    if st.sidebar.button('🚀 啟動模型訓練'):
        if st.session_state.get('train_signature') == signature and 'model' in st.session_state:
            st.sidebar.success('條件一致，已直接使用目前記憶體中的模型。')
            return

        cached_state = _load_cached_state(signature)
        if cached_state is not None:
            st.session_state.update(cached_state)
            st.session_state['train_signature'] = signature
            st.sidebar.success('條件一致，已載入已儲存模型。')
            return

        with st.spinner('AI 運算中 (模型訓練中)...'):
            try:
                state = train_model(df_raw, dye_cols, model_type=model_type, model_params=model_params)
            except ImportError as exc:
                st.sidebar.error(str(exc))
                return
            st.session_state.update(state)
            st.session_state['train_signature'] = signature
            _save_cached_state(signature, state)
            st.sidebar.success(f"訓練完成：{state['model_label']}")
            if state.get('automl_info'):
                st.sidebar.info(f"最佳CV分數: {state['automl_info']['best_cv_score']:.4f}")
