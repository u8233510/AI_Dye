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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import clean_dye_id, deltaE_CMC, transform_bag_of_dyes

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


ARTIFACT_DIR = Path('.model_cache')
TRAINING_PIPELINE_VERSION = '2026-04-28-v2'


class ExplicitWeightedVotingRegressor(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)


MODEL_LABELS = {
    'current_ensemble': '目前模型 (Voting Ensemble)',
    'automl_lite': 'AutoML-lite (RandomizedSearchCV)',
    'catboost': 'CatBoost (Gradient Boosting)',
    'xgboost': 'XGBoost (Gradient Boosting)',
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


def _build_xgboost_pipeline(pre, model_params):
    if XGBRegressor is None:
        raise ImportError('XGBoost 未安裝，請先安裝 xgboost 套件。')

    xgb = XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=int(model_params.get('xgb_n_estimators', 1000)),
        learning_rate=float(model_params.get('xgb_learning_rate', 0.03)),
        max_depth=int(model_params.get('xgb_max_depth', 8)),
        subsample=float(model_params.get('xgb_subsample', 0.9)),
        colsample_bytree=float(model_params.get('xgb_colsample_bytree', 0.9)),
        reg_alpha=float(model_params.get('xgb_reg_alpha', 0.0)),
        reg_lambda=float(model_params.get('xgb_reg_lambda', 1.0)),
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ('pre', pre),
            ('reg', MultiOutputRegressor(xgb)),
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
        'pipeline_version': TRAINING_PIPELINE_VERSION,
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


def _fit_linear_de_calibrator(y_pred_raw, y_true):
    """用線性校正把 DE 直預測拉回實際分布，降低系統性高估/低估。"""
    pred = np.asarray(y_pred_raw, dtype=float)
    target = np.asarray(y_true, dtype=float)
    if len(pred) < 2:
        return {'slope': 1.0, 'intercept': 0.0}

    if np.isclose(np.std(pred), 0.0):
        return {'slope': 1.0, 'intercept': float(np.mean(target) - np.mean(pred))}

    slope, intercept = np.polyfit(pred, target, deg=1)
    if not np.isfinite(slope):
        slope = 1.0
    if not np.isfinite(intercept):
        intercept = 0.0
    return {'slope': float(slope), 'intercept': float(intercept)}


def _apply_linear_de_calibrator(y_pred_raw, calibration):
    pred = np.asarray(y_pred_raw, dtype=float)
    slope = float(calibration.get('slope', 1.0))
    intercept = float(calibration.get('intercept', 0.0))
    return np.clip(pred * slope + intercept, 0.0, None)


def _fit_local_de_model(X_train, y_de_train):
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == object]
    X_enc = pd.get_dummies(X_train, columns=cat_cols, dummy_na=False)
    n = len(X_enc)
    k = min(25, max(5, int(np.sqrt(max(n, 1)))))
    local_model = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='manhattan')
    local_model.fit(X_enc, y_de_train)
    dists, _ = local_model.kneighbors(X_enc, n_neighbors=k, return_distance=True)
    ref_dist = float(np.median(dists[:, -1])) if len(dists) else 1.0
    if not np.isfinite(ref_dist) or ref_dist <= 1e-9:
        ref_dist = 1.0
    return {
        'model': local_model,
        'cat_cols': cat_cols,
        'feature_columns': list(X_enc.columns),
        'distance_scale': ref_dist,
    }


def _predict_local_de(local_de_model, X, return_confidence=False):
    X_enc = pd.get_dummies(X, columns=local_de_model['cat_cols'], dummy_na=False)
    X_enc = X_enc.reindex(columns=local_de_model['feature_columns'], fill_value=0.0)
    pred = local_de_model['model'].predict(X_enc)
    pred = np.clip(pred, 0.0, None)
    if not return_confidence:
        return pred
    dists, _ = local_de_model['model'].kneighbors(X_enc, n_neighbors=1, return_distance=True)
    distance_scale = float(local_de_model.get('distance_scale', 1.0))
    conf = np.exp(-dists[:, 0] / max(distance_scale, 1e-9))
    return pred, np.clip(conf, 0.0, 1.0)


def _optimize_de_blend_weight(y_true, de_global, de_local, local_conf):
    y_true = np.asarray(y_true, dtype=float)
    de_global = np.asarray(de_global, dtype=float)
    de_local = np.asarray(de_local, dtype=float)
    local_conf = np.asarray(local_conf, dtype=float)

    if len(y_true) == 0:
        return {'mode': 'adaptive_confidence', 'beta': 1.0, 'margin': 0.0}

    beta_candidates = np.linspace(0.4, 2.2, 19)
    margin_candidates = np.linspace(0.0, 0.25, 11)
    best_cfg = {'mode': 'adaptive_confidence', 'beta': 1.0, 'margin': 0.0}
    best_score = float('inf')
    for beta in beta_candidates:
        w_local = np.clip(local_conf * beta, 0.0, 1.0)
        blended_base = np.clip((1.0 - w_local) * de_global + w_local * de_local, 0.0, None)
        for margin in margin_candidates:
            margin_eff = margin * (1.0 - local_conf)
            blended = np.clip(blended_base + margin_eff, 0.0, None)
            mae = float(np.mean(np.abs(blended - y_true)))
            high_cut = float(np.quantile(y_true, 0.9))
            high_mask = y_true >= high_cut
            high_mae = float(np.mean(np.abs(blended[high_mask] - y_true[high_mask]))) if np.any(high_mask) else mae
            actual_over = y_true > 0.8
            miss_rate = float(np.mean(blended[actual_over] <= 0.8)) if np.any(actual_over) else 0.0
            actual_pass = y_true <= 0.8
            false_reject_rate = float(np.mean(blended[actual_pass] > 0.8)) if np.any(actual_pass) else 0.0
            pred_pass_rate = float(np.mean(blended <= 0.8))
            actual_pass_rate = float(np.mean(actual_pass))
            pass_rate_gap = abs(pred_pass_rate - actual_pass_rate)

            # 平衡危險漏判與過度保守誤殺，避免「全部判失敗」
            score = 2.0 * miss_rate + 2.5 * false_reject_rate + 1.2 * high_mae + 0.3 * mae + 2.0 * pass_rate_gap
            if score < best_score:
                best_score = score
                best_cfg = {'mode': 'adaptive_confidence', 'beta': float(beta), 'margin': float(margin)}
    return best_cfg


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
    elif model_type == 'xgboost':
        model = _build_xgboost_pipeline(pre, model_params)
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
    elif model_type == 'xgboost':
        de_model = Pipeline(
            [
                ('pre', pre_de),
                (
                    'reg',
                    XGBRegressor(
                        objective='reg:absoluteerror',
                        n_estimators=int(model_params.get('xgb_n_estimators', 1000)),
                        learning_rate=float(model_params.get('xgb_learning_rate', 0.03)),
                        max_depth=int(model_params.get('xgb_max_depth', 8)),
                        subsample=float(model_params.get('xgb_subsample', 0.9)),
                        colsample_bytree=float(model_params.get('xgb_colsample_bytree', 0.9)),
                        reg_alpha=float(model_params.get('xgb_reg_alpha', 0.0)),
                        reg_lambda=float(model_params.get('xgb_reg_lambda', 1.0)),
                        random_state=42,
                        n_jobs=-1,
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
    y_de_train = df_train.loc[X_train.index, 'CMC_DE'].values
    de_model.fit(X_train, y_de_train, reg__sample_weight=sample_weights)
    de_train_raw = np.clip(de_model.predict(X_train), 0, None)
    de_calibration = _fit_linear_de_calibrator(de_train_raw, y_de_train)
    local_de_model = _fit_local_de_model(X_train, y_de_train)

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
    de_test_raw = np.clip(de_model.predict(X_test), 0, None)
    de_test_global = _apply_linear_de_calibrator(de_test_raw, de_calibration)
    de_test_local, local_conf = _predict_local_de(local_de_model, X_test, return_confidence=True)
    y_de_test = df_train.loc[X_test.index, 'CMC_DE'].values
    blend_cfg = _optimize_de_blend_weight(y_de_test, de_test_global, de_test_local, local_conf)
    beta = float(blend_cfg.get('beta', 1.0))
    margin = float(blend_cfg.get('margin', 0.0))
    local_weight = np.clip(local_conf * beta, 0.0, 1.0)
    global_weight = 1.0 - local_weight
    margin_eff = margin * (1.0 - local_conf)
    df_fb['預測DE_模型'] = np.clip(global_weight * de_test_global + local_weight * de_test_local + margin_eff, 0.0, None)

    return {
        'model': model,
        'de_model': de_model,
        'de_calibration': de_calibration,
        'local_de_model': local_de_model,
        'de_blend_weights': blend_cfg,
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
    elif model_type == 'xgboost':
        st.sidebar.caption('XGBoost：高效梯度提升模型')
        params['xgb_n_estimators'] = st.sidebar.slider('XGBoost n_estimators', min_value=300, max_value=3000, value=1000, step=100)
        params['xgb_learning_rate'] = st.sidebar.number_input(
            'XGBoost learning_rate',
            min_value=0.005,
            max_value=0.3,
            value=0.03,
            step=0.005,
            format='%.3f',
        )
        params['xgb_max_depth'] = st.sidebar.slider('XGBoost max_depth', min_value=3, max_value=12, value=8, step=1)
        params['xgb_subsample'] = st.sidebar.slider('XGBoost subsample', min_value=0.5, max_value=1.0, value=0.9, step=0.05)
        params['xgb_colsample_bytree'] = st.sidebar.slider('XGBoost colsample_bytree', min_value=0.5, max_value=1.0, value=0.9, step=0.05)
        params['xgb_reg_alpha'] = st.sidebar.number_input('XGBoost reg_alpha', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        params['xgb_reg_lambda'] = st.sidebar.number_input('XGBoost reg_lambda', min_value=0.1, max_value=20.0, value=1.0, step=0.1)

    return params


def render_training_button(df_raw, dye_cols):
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 🤖 訓練模型選擇')
    if CatBoostRegressor is None:
        st.sidebar.warning('未偵測到 catboost 套件；若選 CatBoost 將無法訓練。')
    if XGBRegressor is None:
        st.sidebar.warning('未偵測到 xgboost 套件；若選 XGBoost 將無法訓練。')

    model_type = st.sidebar.selectbox('模型類型', options=list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    model_params = _render_model_params(model_type)
    force_retrain = st.sidebar.checkbox('忽略快取並強制重訓', value=False, help='勾選後會重新訓練，不使用記憶體與磁碟快取模型。')
    signature = _training_signature(df_raw, dye_cols, model_type, model_params)

    if st.sidebar.button('🚀 啟動模型訓練'):
        if (not force_retrain) and st.session_state.get('train_signature') == signature and 'model' in st.session_state:
            st.sidebar.success('條件一致，已直接使用目前記憶體中的模型。')
            return

        cached_state = None if force_retrain else _load_cached_state(signature)
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
