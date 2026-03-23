# ============================================================
# FINAL PIPELINE 
# CLEAN + FEATURE ENGINEERING + CATBOOST + RIDGE ENSEMBLE
# + TIME-BASED VALIDATION + BUSINESS CLIP + SHAP + LIME
# ============================================================

import os
import numpy as np
import pandas as pd
import catboost as cb

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================
# 0) CONFIG
# ============================================================
SEED = 42

RUN_EXPLAIN = True
N_SHAP_SAMPLE = 2000
N_LIME_FEATURES = 12

OUT_SUBMISSION = "submission_final_ensemble.csv"

DATA_DIR = r"C:\Users\T480s\Downloads"
PATH_ADMISSION = os.path.join(DATA_DIR, "admission.csv")
PATH_ACADEMIC  = os.path.join(DATA_DIR, "academic_records.csv")
PATH_TEST      = os.path.join(DATA_DIR, "test.csv")
PATH_SAMPLE    = os.path.join(DATA_DIR, "sample_submission.csv")


# ============================================================
# 1) UTILITIES
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    return df


def parse_semester(sem_str: str):
    """
    Input example: "HK1 2022-2023"
    Return: (start_year, hk_num, sem_id)
    """
    try:
        hk, yr = sem_str.split(" ")
        hk_num = int(hk.replace("HK", "").strip())
        start_year = int(yr.split("-")[0])
        return start_year, hk_num, start_year * 10 + hk_num
    except:
        return 0, 0, 0


def zscore_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    g = df.groupby(group_col)[value_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1)
    return (df[value_col] - mean) / std


def safe_fillna_all(df: pd.DataFrame, num_fill=-1, cat_fill="-1") -> pd.DataFrame:
    """
    Fill NA consistently:
    - numeric -> num_fill
    - non-numeric -> cat_fill
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(num_fill)
        else:
            df[col] = df[col].fillna(cat_fill)
    return df


# ============================================================
# 2) LOAD DATA
# ============================================================
admission = pd.read_csv(PATH_ADMISSION, dtype={"MA_SO_SV": str})
academic  = pd.read_csv(PATH_ACADEMIC, dtype={"MA_SO_SV": str})
test      = pd.read_csv(PATH_TEST, dtype={"MA_SO_SV": str})
sample    = pd.read_csv(PATH_SAMPLE, dtype={"MA_SO_SV": str})

admission = clean_columns(admission)
academic  = clean_columns(academic)
test      = clean_columns(test)
sample    = clean_columns(sample)


# ============================================================
# 3) PREPROCESS - ADMISSION
# ============================================================
admission = admission.drop_duplicates(subset=["MA_SO_SV"], keep="last").reset_index(drop=True)
admission["PTXT"] = admission["PTXT"].astype(str)

# Fill DIEM_TRUNGTUYEN = 0 using defendable median strategy
global_median = admission.loc[admission["DIEM_TRUNGTUYEN"] > 0, "DIEM_TRUNGTUYEN"].median()
ptxt_500_median = admission.loc[
    (admission["PTXT"] == "500") & (admission["DIEM_TRUNGTUYEN"] > 0),
    "DIEM_TRUNGTUYEN"
].median()

mask_ptxt500 = (admission["PTXT"] == "500") & (admission["DIEM_TRUNGTUYEN"] == 0)
admission.loc[mask_ptxt500, ["DIEM_TRUNGTUYEN", "DIEM_CHUAN"]] = ptxt_500_median

mask_ptxt303 = (admission["PTXT"] == "303") & (admission["DIEM_TRUNGTUYEN"] == 0)
admission.loc[mask_ptxt303, ["DIEM_TRUNGTUYEN", "DIEM_CHUAN"]] = global_median

# Normalize score within PTXT group
admission["DIEM_TRUNGTUYEN_SCALED"] = (
    zscore_by_group(admission, "PTXT", "DIEM_TRUNGTUYEN").fillna(0)
)


# ============================================================
# 4) PREPROCESS - ACADEMIC
# ============================================================
academic = academic.drop_duplicates(subset=["MA_SO_SV", "HOC_KY"], keep="last").reset_index(drop=True)

# Clip GPA/CPA to valid bounds
academic["GPA"] = academic["GPA"].clip(lower=0, upper=4.0)
academic["CPA"] = academic["CPA"].clip(lower=0, upper=4.0)

# Parse semester
academic[["start_year", "hk_num", "sem_id"]] = academic["HOC_KY"].apply(
    lambda x: pd.Series(parse_semester(x))
)
academic = academic.sort_values(["MA_SO_SV", "sem_id"]).reset_index(drop=True)

# Forward-fill CPA per student (0 treated as missing)
academic["CPA"] = academic["CPA"].replace(0, np.nan)
academic["CPA"] = academic.groupby("MA_SO_SV")["CPA"].ffill()
academic["CPA"] = academic["CPA"].fillna(academic["GPA"]).clip(upper=4.0)

# -------------------------
# FLAGS (Defendable)
# -------------------------
# Case: GPA>0 but completed credits=0 => can happen in reality -> keep, only flag
academic["FLAG_GPA_POS_TC0"] = ((academic["GPA"] > 0) & (academic["TC_HOANTHANH"] == 0)).astype(int)

# Case: GPA=0 but completed credits>0 => likely inconsistent -> flag + optional sanitize
academic["FLAG_GPA0_TC_POS"] = ((academic["GPA"] == 0) & (academic["TC_HOANTHANH"] > 0)).astype(int)
academic.loc[academic["FLAG_GPA0_TC_POS"] == 1, "GPA"] = np.nan

# Compatibility flag (if you want a single combined flag)
academic["FLAG_INCONSISTENT"] = academic["FLAG_GPA_POS_TC0"].astype(int)

GLOBAL_MEAN_GPA = academic["GPA"].mean(skipna=True)


# ============================================================
# 5) FEATURE ENGINEERING (History features)
# ============================================================
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("MA_SO_SV", sort=False)

    df["n_prev_sem"] = g.cumcount()

    df["lag1_GPA"] = g["GPA"].shift(1)
    df["lag1_TC_HOANTHANH"] = g["TC_HOANTHANH"].shift(1)
    df["lag1_TC_DANGKY"] = g["TC_DANGKY"].shift(1)

    df["avg_hist_GPA"] = g["GPA"].shift(1).expanding().mean().reset_index(0, drop=True)
    df["avg_hist_TC_HOANTHANH"] = g["TC_HOANTHANH"].shift(1).expanding().mean().reset_index(0, drop=True)

    df["GPA_trend"] = df["lag1_GPA"] - df["avg_hist_GPA"]
    df["TC_trend"] = df["lag1_TC_HOANTHANH"] - df["avg_hist_TC_HOANTHANH"]

    df["roll3_GPA_std"] = g["GPA"].shift(1).rolling(3, min_periods=2).std().reset_index(0, drop=True)

    return df


acad_feat = add_history_features(academic)


# ============================================================
# 6) TRAIN/VALID SPLIT (time-based)
# ============================================================
train_mask = (acad_feat["start_year"] < 2023) | ((acad_feat["start_year"] == 2023) & (acad_feat["hk_num"] == 1))
valid_mask = (acad_feat["start_year"] == 2023) & (acad_feat["hk_num"] == 2)


# ============================================================
# 7) FEATURE LISTS
# ============================================================
NUM_FEATURES = [
    "TC_DANGKY",
    "lag1_GPA", "avg_hist_GPA", "GPA_trend",
    "lag1_TC_HOANTHANH", "avg_hist_TC_HOANTHANH", "TC_trend",
    "n_prev_sem",
    "DIEM_TRUNGTUYEN_SCALED",
    "roll3_GPA_std",
    "FLAG_INCONSISTENT",
    "FLAG_GPA0_TC_POS",
    "FLAG_GPA_POS_TC0"
]
CAT_FEATURES = ["PTXT", "TOHOP_XT"]


def compute_workload_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Workload pressure proxy:
    ratio = TC_DANGKY / (avg_hist_TC_HOANTHANH + 1)
    With smoothing for early semesters + log1p + clip
    """
    raw_ratio = df["TC_DANGKY"] / (df["avg_hist_TC_HOANTHANH"] + 1)

    ratio = np.where(
        df["n_prev_sem"] == 0,
        1.0,
        np.where(df["n_prev_sem"] == 1, np.sqrt(raw_ratio), raw_ratio)
    )

    ratio = np.log1p(ratio)
    ratio = np.clip(ratio, 0, 2)
    return ratio


def make_model_dataset(acad_df: pd.DataFrame, admission_df: pd.DataFrame):
    """
    Merge + compute derived features + fill NA
    Return:
        X, y, tc_dangky
    """
    df = acad_df.merge(admission_df, on="MA_SO_SV", how="left")

    df["Workload_Ratio"] = compute_workload_ratio(df)

    # Fill missing
    df = safe_fillna_all(df, num_fill=-1, cat_fill="-1")

    # Ensure categorical as str
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)

    X = df[NUM_FEATURES + CAT_FEATURES + ["Workload_Ratio"]].copy()
    y = df["TC_HOANTHANH"].astype(float)
    tc_dangky = df["TC_DANGKY"].astype(float)

    return X, y, tc_dangky


X_train, y_train, tc_train = make_model_dataset(acad_feat.loc[train_mask].copy(), admission)
X_valid, y_valid, tc_valid = make_model_dataset(acad_feat.loc[valid_mask].copy(), admission)

cat_indices = [X_train.columns.get_loc(c) for c in CAT_FEATURES]


# ============================================================
# 8) TRAIN MODELS
# ============================================================
cb_model = cb.CatBoostRegressor(
    iterations=20000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=10,
    random_seed=SEED,
    verbose=300,
    loss_function="RMSE",
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=500
)

cb_model.fit(
    X_train, y_train,
    cat_features=cat_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES + ["Workload_Ratio"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
])

ridge_model = Pipeline([
    ("prep", preprocessor),
    ("model", Ridge(alpha=2.0, random_state=SEED))
])

ridge_model.fit(X_train, y_train)


# ============================================================
# 9) VALIDATION + BEST BLEND
# ============================================================
val_pred_cb = cb_model.predict(X_valid)
val_pred_ridge = ridge_model.predict(X_valid)

best_catboost_weight, best_rmse = None, float("inf")

for w in np.linspace(0, 1, 51):
    pred = w * val_pred_cb + (1 - w) * val_pred_ridge
    pred = np.clip(pred, 0, tc_valid)  # business constraint
    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_catboost_weight = w

print(f"\n[VALID] Best blend weight (CatBoost): {best_catboost_weight:.2f}")
print(f"[VALID] RMSE: {best_rmse:.5f}")


# ============================================================
# 10) TEST PREDICT + SUBMISSION
# ============================================================
# Use last record per student to infer history-based features
last_record = acad_feat.groupby("MA_SO_SV").tail(1).copy()

test_proc = (
    test.merge(
        last_record.drop(columns=["GPA", "TC_HOANTHANH", "TC_DANGKY", "HOC_KY"], errors="ignore"),
        on="MA_SO_SV",
        how="left"
    )
    .merge(admission, on="MA_SO_SV", how="left")
)

test_proc["Workload_Ratio"] = compute_workload_ratio(test_proc)

# Strong defaults for cold-start students
DEFAULTS = {
    "lag1_GPA": GLOBAL_MEAN_GPA,
    "avg_hist_GPA": GLOBAL_MEAN_GPA,
    "lag1_TC_HOANTHANH": 15,
    "avg_hist_TC_HOANTHANH": 15,
    "GPA_trend": 0,
    "TC_trend": 0,
    "n_prev_sem": 0,
    "roll3_GPA_std": 0,
    "FLAG_INCONSISTENT": 0,
    "FLAG_GPA0_TC_POS": 0,
    "FLAG_GPA_POS_TC0": 0,
    "Workload_Ratio": 1.0,
    "DIEM_TRUNGTUYEN_SCALED": 0
}

for col, val in DEFAULTS.items():
    if col in test_proc.columns:
        test_proc[col] = test_proc[col].fillna(val)

test_proc = safe_fillna_all(test_proc, num_fill=-1, cat_fill="-1")
for c in CAT_FEATURES:
    test_proc[c] = test_proc[c].astype(str)

X_test = test_proc[NUM_FEATURES + CAT_FEATURES + ["Workload_Ratio"]]

pred_cb = cb_model.predict(X_test)
pred_ridge = ridge_model.predict(X_test)

pred_ensemble = best_catboost_weight * pred_cb + (1 - best_catboost_weight) * pred_ridge
pred_ensemble = np.clip(pred_ensemble, 0, test_proc["TC_DANGKY"].astype(float))  # business clip

submission = sample.copy()
target_col = [c for c in submission.columns if c != "MA_SO_SV"][0]
submission[target_col] = pred_ensemble.astype(float)

submission.to_csv(OUT_SUBMISSION, index=False)
print(f"\nSaved: {OUT_SUBMISSION}")


# ============================================================
# 11) EXPLAINABILITY (SHAP + LIME)
# ============================================================
if RUN_EXPLAIN:
    import matplotlib.pyplot as plt
    import shap

    # ---------- SHAP (CatBoost) ----------
    try:
        X_shap = X_train.sample(n=min(N_SHAP_SAMPLE, len(X_train)), random_state=SEED)

        explainer = shap.TreeExplainer(cb_model)
        shap_values = explainer.shap_values(X_shap)

        plt.figure()
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.title("SHAP Summary - CatBoost")
        plt.tight_layout()
        plt.savefig("shap_summary_catboost.png", dpi=200)
        plt.show()

        plt.figure()
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar) - CatBoost")
        plt.tight_layout()
        plt.savefig("shap_bar_catboost.png", dpi=200)
        plt.show()

        idx = 0
        x_one = X_valid.iloc[[idx]]
        shap_one = explainer.shap_values(x_one)

        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_one[0],
                base_values=explainer.expected_value,
                data=x_one.iloc[0],
                feature_names=x_one.columns
            ),
            show=False
        )
        plt.title("SHAP Waterfall - 1 Student (CatBoost)")
        plt.tight_layout()
        plt.savefig("shap_waterfall_one_student.png", dpi=200)
        plt.show()

        print("Saved SHAP plots: shap_summary_catboost.png, shap_bar_catboost.png, shap_waterfall_one_student.png")

    except Exception as e:
        print("[WARN] SHAP failed:", e)

    # ---------- LIME (Ridge) ----------
    try:
        from lime.lime_tabular import LimeTabularExplainer

        X_train_lime = X_train.copy()
        X_valid_lime = X_valid.copy()

        lime_cat_idx = [X_train_lime.columns.get_loc(c) for c in CAT_FEATURES]

        explainer_lime = LimeTabularExplainer(
            training_data=X_train_lime.values,
            feature_names=X_train_lime.columns.tolist(),
            categorical_features=lime_cat_idx,
            mode="regression",
            discretize_continuous=True,
            random_state=SEED
        )

        i = 0
        x_instance = X_valid_lime.iloc[i].values

        exp = explainer_lime.explain_instance(
            data_row=x_instance,
            predict_fn=ridge_model.predict,
            num_features=N_LIME_FEATURES
        )

        lime_path = "lime_ridge_local_explain.html"
        exp.save_to_file(lime_path)
        print(f"Saved: {lime_path}")

    except Exception as e:
        print("[WARN] LIME failed:", e)