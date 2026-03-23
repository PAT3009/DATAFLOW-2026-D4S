# ============================================================
# FINAL PIPELINE - CLEAN + FEATURE ENGINEERING + CATBOOST ENSEMBLE
# + RMSE VALIDATION + BUSINESS-CONSTRAINT CLIP + SHAP + LIME
# (Stable, Defendable, No Hack) - UPDATED with FIX(2) + FIX(3)
# ============================================================


import os
import numpy as np
import pandas as pd
import catboost as cb


from sklearn.metrics import mean_squared_error


# -------------------------
# CONFIG
# -------------------------
SEED = 42
RUN_EXPLAIN = True          # bật/tắt SHAP + LIME
N_SHAP_SAMPLE = 2000        # sample size để chạy SHAP nhanh
N_LIME_FEATURES = 12        # số feature show trong LIME


OUT_SUBMISSION = "submission_final_ensemble.csv"


# ============================================================
# 1) LOAD DATA + CLEAN COLUMNS
# ============================================================


admission = pd.read_csv(r'\admission.csv', dtype={'MA_SO_SV': str})
academic  = pd.read_csv(r'\academic_records.csv', dtype={'MA_SO_SV': str})
test      = pd.read_csv(r'\test.csv', dtype={'MA_SO_SV': str})
sample    = pd.read_csv(r'\sample_submission.csv', dtype={'MA_SO_SV': str})




def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    return df




admission = clean_columns(admission)
academic  = clean_columns(academic)
test      = clean_columns(test)
sample    = clean_columns(sample)


# ============================================================
# 2) ADVANCED CLEANING & PREPROCESSING
# ============================================================


# ---------- A) ADMISSION ----------
admission = admission.drop_duplicates(subset=["MA_SO_SV"], keep="last").reset_index(drop=True)
admission["PTXT"] = admission["PTXT"].astype(str)


# fill DIEM_TRUNGTUYEN = 0 theo median (defendable)
global_median = admission.loc[admission["DIEM_TRUNGTUYEN"] > 0, "DIEM_TRUNGTUYEN"].median()
ptxt_500_median = admission.loc[
    (admission["PTXT"] == "500") & (admission["DIEM_TRUNGTUYEN"] > 0),
    "DIEM_TRUNGTUYEN"
].median()


mask_500 = (admission["PTXT"] == "500") & (admission["DIEM_TRUNGTUYEN"] == 0)
admission.loc[mask_500, ["DIEM_TRUNGTUYEN", "DIEM_CHUAN"]] = ptxt_500_median


mask_303 = (admission["PTXT"] == "303") & (admission["DIEM_TRUNGTUYEN"] == 0)
admission.loc[mask_303, ["DIEM_TRUNGTUYEN", "DIEM_CHUAN"]] = global_median




def scale_by_group(data: pd.DataFrame, group_col: str, target_col: str) -> pd.Series:
    g = data.groupby(group_col)[target_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1)
    return (data[target_col] - mean) / std




admission["DIEM_TRUNGTUYEN_SCALED"] = scale_by_group(admission, "PTXT", "DIEM_TRUNGTUYEN").fillna(0)


# ---------- B) ACADEMIC ----------
academic = academic.drop_duplicates(subset=["MA_SO_SV", "HOC_KY"], keep="last").reset_index(drop=True)


# clip GPA/CPA
academic["GPA"] = academic["GPA"].clip(lower=0, upper=4.0)
academic["CPA"] = academic["CPA"].clip(lower=0, upper=4.0)


# parse semester
def parse_hocky(sem_str):
    try:
        hk, yr = sem_str.split(" ")
        hk_num = int(hk.replace("HK", "").strip())
        start_year = int(yr.split("-")[0])
        return start_year, hk_num, start_year * 10 + hk_num
    except:
        return 0, 0, 0




academic[["start_year", "hk_num", "sem_id"]] = academic["HOC_KY"].apply(lambda x: pd.Series(parse_hocky(x)))
academic = academic.sort_values(["MA_SO_SV", "sem_id"]).reset_index(drop=True)


# CPA forward fill per student
academic["CPA"] = academic["CPA"].replace(0, np.nan)
academic["CPA"] = academic.groupby("MA_SO_SV")["CPA"].ffill()
academic["CPA"] = academic["CPA"].fillna(academic["GPA"]).clip(upper=4.0)


academic["TC_HOANTHANH_RAW"] = academic["TC_HOANTHANH"]


# ============================================================
# FIX DATA INCONSISTENCY
# ============================================================


# Case 1: GPA>0 nhưng TC_HOANTHANH=0 => bất thường
mask_incon1 = (academic["GPA"] > 0) & (academic["TC_HOANTHANH"] == 0)
academic["FLAG_INCONSISTENT"] = mask_incon1.astype(int)


# Case 2: GPA=0 nhưng TC_HOANTHANH>0 => bất thường
mask_incon2 = (academic["GPA"] == 0) & (academic["TC_HOANTHANH"] > 0)
academic["FLAG_INCONSISTENT_2"] = mask_incon2.astype(int)


# sanitize nhẹ: GPA=0 trong case này có thể missing
academic.loc[mask_incon2, "GPA"] = np.nan


GLOBAL_MEAN_GPA = academic["GPA"].mean(skipna=True)


# ============================================================
# 3) FEATURE ENGINEERING (history features)
# ============================================================


def generate_history_features(df: pd.DataFrame) -> pd.DataFrame:
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




academic_ext = generate_history_features(academic)


academic_ext["PASS_RATIO"] = (
    academic_ext["lag1_TC_HOANTHANH"] /
    (academic_ext["lag1_TC_DANGKY"] + 1e-6)
).clip(0, 1.2)

