
import pandas as pd
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASE_DIR = r"C:\Users\Volkov-iv\Desktop\5minModel-py"
train_dir = os.path.join(BASE_DIR, "compressor_train")
bound_dir = os.path.join(BASE_DIR, "compressor_bound")
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)

def parse_power_series(s):
    if pd.isna(s) or s == "":
        return np.array([])
    return np.array([float(x) for x in str(s).replace(" ", "").split(",")])


def get_bounds_from_json(temp, TEMP_BOUNDS):
    key = str(round(float(temp), 1))
    if key not in TEMP_BOUNDS:
        return np.nan, np.nan
    return float(TEMP_BOUNDS[key][0]), float(TEMP_BOUNDS[key][1])

#  извлечение фич
def extract_power_features(row, TEMP_BOUNDS):
    power = parse_power_series(row["power"])
    temp = row["temperature"]

    if len(power) == 0:
        return pd.Series({})

    bmin, bmax = get_bounds_from_json(temp, TEMP_BOUNDS)

    if np.isnan(bmin) or np.isnan(bmax) or bmax <= bmin:
        return pd.Series({})

    n = len(power)
    width = (bmax - bmin) + 1e-6

    p_mean = power.mean()
    p_std = power.std()
    p_min = power.min()
    p_max = power.max()

    # отклонения 
    under = np.maximum(bmin - power, 0) / width
    over  = np.maximum(power - bmax, 0) / width

    feats = {}

    #интенсивност
    feats["under_mean"] = under.mean()
    feats["under_max"]  = under.max()
    feats["over_mean"]  = over.mean()
    feats["over_max"]   = over.max()

    #частота
    feats["under_frac"] = np.mean(power < bmin)
    feats["over_frac"]  = np.mean(power > bmax)

    # положение среднего
    feats["mean_pos"] = (p_mean - bmin) / width

    #запас до границ
    feats["margin_to_min"] = (p_min - bmin) / width
    feats["margin_to_max"] = (bmax - p_max) / width

    #динамика 
    feats["max_jump"] = np.max(np.abs(np.diff(power))) / width if n > 1 else 0.0

    if n > 1:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, power, 1)
        trend = slope * x + intercept
    else:
        slope = 0.0

    feats["trend_slope"] = slope / width

    # нестабильность
    feats["power_cv"] = p_std / (p_mean + 1e-6)

    return pd.Series(feats)


FEATURE_ORDER = [
    "under_mean",
    "under_max",
    "over_mean",
    "over_max",
    "under_frac",
    "over_frac",
    "mean_pos",
    "margin_to_min",
    "margin_to_max",
    "max_jump",
    "trend_slope",
    "power_cv",
]

for filename in os.listdir(train_dir):
    if not filename.endswith("_train.csv"):
        continue

    model_name = filename.replace("_train.csv", "")
    print(f"\n================ {model_name} ================")

    train_path = os.path.join(train_dir, filename)
    bound_path = os.path.join(bound_dir, f"{model_name}_boundaries.json")

    if not os.path.exists(bound_path):
        print("No boundaries:", bound_path)
        continue

    with open(bound_path) as f:
        TEMP_BOUNDS = json.load(f)

    df = pd.read_csv(train_path)

    print("Rows:", len(df))
    print(df["label"].value_counts())

    X_full = df.apply(lambda r: extract_power_features(r, TEMP_BOUNDS), axis=1)
    X = X_full.reindex(columns=FEATURE_ORDER).fillna(0.0)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=6.0,
        reg_alpha=1.5,
        min_child_weight=10,
        gamma=0.5,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("ROC AUC:", auc)

    booster = model.get_booster()
    config = json.loads(booster.save_config())

    raw_base = config["learner"]["learner_model_param"]["base_score"]
    raw_base = raw_base.strip("[]")
    base_score = float(raw_base)

    base_logit = np.log(base_score / (1 - base_score))

    print(f"BASE SCORE: {base_score:.6f}")
    print(f"BASE LOGIT : {base_logit:.6f}")

    # bias модели
    y_margin = model.predict(X_train, output_margin=True)
    real_bias = float(np.mean(y_margin))

    print(f" биас: {real_bias:.6f}")
    
    booster.set_attr(boundary_table=json.dumps(TEMP_BOUNDS))

    model_path = os.path.join(model_dir, f"{model_name}_model.json")
    booster.save_model(model_path)

    print("Готово:", model_path)


