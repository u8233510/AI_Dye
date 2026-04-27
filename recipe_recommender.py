import numpy as np
import pandas as pd


def calculate_lab_and_deltas(std_l, std_a, std_b, pred_dl, pred_dc, pred_dh):
    """將模型輸出的 DL/DC/Dh 還原成 L/a/b 與 DL/Da/Db。"""
    std_c = np.sqrt(std_a**2 + std_b**2)
    std_h = np.degrees(np.arctan2(std_b, std_a)) % 360

    pred_c = max(std_c + pred_dc, 0)
    pred_h = (std_h + pred_dh) % 360

    pred_l = std_l + pred_dl
    pred_a = pred_c * np.cos(np.radians(pred_h))
    pred_b = pred_c * np.sin(np.radians(pred_h))

    pred_da = pred_a - std_a
    pred_db = pred_b - std_b
    return pred_l, pred_a, pred_b, pred_dl, pred_da, pred_db


def build_dye_bounds(df_raw, dye_cols, known_dyes, clean_dye_id_fn):
    """從歷史資料抓每一支染料可用濃度上界，供推薦時取樣。"""
    bounds = {d: (0.0, 5.0) for d in known_dyes}

    for d in known_dyes:
        values = []
        for d_col in dye_cols:
            c_col = d_col.replace('料號', '濃度')
            if c_col not in df_raw.columns:
                continue
            mask = df_raw[d_col].apply(clean_dye_id_fn) == d
            if mask.any():
                values.extend(df_raw.loc[mask, c_col].dropna().astype(float).tolist())

        if values:
            max_v = float(np.percentile(values, 95))
            bounds[d] = (0.0, max(0.01, max_v))

    return bounds


def recommend_recipe(
    model,
    transform_bag_of_dyes_fn,
    deltae_cmc_fn,
    known_dyes,
    dye_cols,
    std_l,
    std_a,
    std_b,
    dpf,
    op,
    shade_name,
    shade_id,
    selected_dyes,
    dye_bounds,
    n_samples=1500,
    random_state=42,
):
    """隨機搜尋推薦配方濃度，目標是最小化預測 CMC DE。"""
    rng = np.random.default_rng(random_state)

    selected_dyes = [d for d in selected_dyes if d in known_dyes]
    if not selected_dyes:
        return None

    best = None

    for _ in range(n_samples):
        candidate_conc = {}
        for d in selected_dyes:
            lo, hi = dye_bounds.get(d, (0.0, 5.0))
            candidate_conc[d] = float(rng.uniform(lo, hi))

        row = {
            '標準樣L': std_l,
            '標準樣a': std_a,
            '標準樣b': std_b,
            'DPF': dpf,
            'OP否': op,
            '色系名稱': shade_name,
            '色系編號': shade_id,
        }

        for i in range(1, 7):
            if i <= len(selected_dyes):
                dye_id = selected_dyes[i - 1]
                row[f'配方料號{i}'] = dye_id
                row[f'配方濃度{i}'] = candidate_conc[dye_id]
            else:
                row[f'配方料號{i}'] = '無'
                row[f'配方濃度{i}'] = 0.0

        df_m = pd.DataFrame([row])
        X_m, _ = transform_bag_of_dyes_fn(df_m, dye_cols, known_dyes=known_dyes)
        pred = model.predict(X_m)[0]

        pred_dl, pred_dc, pred_dh = float(pred[0]), float(pred[1]), float(pred[2])
        pred_l, pred_a, pred_b, pred_dl, pred_da, pred_db = calculate_lab_and_deltas(
            std_l, std_a, std_b, pred_dl, pred_dc, pred_dh
        )
        pred_de = float(deltae_cmc_fn((std_l, std_a, std_b), (pred_l, pred_a, pred_b)))

        if (best is None) or (pred_de < best['CMC_DE']):
            best = {
                'recipe': candidate_conc,
                'L': pred_l,
                'a': pred_a,
                'b': pred_b,
                'DL': pred_dl,
                'Da': pred_da,
                'Db': pred_db,
                'CMC_DE': pred_de,
            }

    return best
