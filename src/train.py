# ============================================================
# 4) TRAIN/VALID SPLIT (time-based)
# ============================================================


train_mask = (academic_ext["start_year"] < 2023) | ((academic_ext["start_year"] == 2023) & (academic_ext["hk_num"] == 1))
valid_mask = (academic_ext["start_year"] == 2023) & (academic_ext["hk_num"] == 2)


X_cols = [
    "TC_DANGKY",
    "lag1_GPA", "avg_hist_GPA", "GPA_trend",
    "lag1_TC_HOANTHANH", "avg_hist_TC_HOANTHANH", "TC_trend",
    "n_prev_sem",
    "DIEM_TRUNGTUYEN_SCALED",
    "roll3_GPA_std",
    "FLAG_INCONSISTENT",
    "FLAG_INCONSISTENT_2",
    "PASS_RATIO"
]
cat_cols = ["PTXT", "TOHOP_XT"]




def build_workload_ratio(df: pd.DataFrame) -> pd.Series:
    raw_ratio = df["TC_DANGKY"] / (df["avg_hist_TC_HOANTHANH"] + 1)


    ratio = np.where(
        df["n_prev_sem"] == 0,
        1.0,
        np.where(df["n_prev_sem"] == 1, np.sqrt(raw_ratio), raw_ratio)
    )


    ratio = np.log1p(ratio)
    ratio = np.clip(ratio, 0, 2)
    return ratio




def finalize_df(acad_ext_df: pd.DataFrame):
    df = acad_ext_df.merge(admission, on="MA_SO_SV", how="left")
    df["Workload_Ratio"] = build_workload_ratio(df)


    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna("-1")


    for c in cat_cols:
        df[c] = df[c].astype(str)


    X = df[X_cols + cat_cols + ["Workload_Ratio"]].copy()
    y = df["TC_HOANTHANH"].astype(float)
    tc_dk = df["TC_DANGKY"].astype(float)


    return X, y, tc_dk




X_train, y_train, tc_train = finalize_df(academic_ext.loc[train_mask].copy())
X_valid, y_valid, tc_valid = finalize_df(academic_ext.loc[valid_mask].copy())


# ============================================================
# SAMPLE WEIGHT (DOWN-WEIGHT NOISY LABELS)
# ============================================================


sample_weight = np.ones(len(y_train))
sample_weight[X_train["FLAG_INCONSISTENT"] == 1] = 0.3


cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]


# ============================================================
# 5) TRAIN MODEL
# ============================================================


cb_model = cb.CatBoostRegressor(
    iterations=8000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=10,
    random_seed=SEED,
    verbose=300,
    loss_function="RMSE",
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=1500
)


cb_model.fit(
    X_train, y_train,
    sample_weight=sample_weight,
    cat_features=cat_indices
)


# ============================================================
# 6) VALID RMSE
# ============================================================


val_pred = cb_model.predict(X_valid)
val_pred = np.clip(val_pred, 0, tc_valid)


y_valid_raw = academic_ext.loc[valid_mask, "TC_HOANTHANH_RAW"].astype(float)
rmse = np.sqrt(mean_squared_error(y_valid_raw, val_pred))
print(f"[VALID] RMSE (CatBoost only): {rmse:.5f}")
