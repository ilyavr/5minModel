import pandas as pd
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


BASE_DIR = r"C:\Users\Volkov-iv\Desktop\simpleFmin"

train_dir = os.path.join(BASE_DIR, "compressor_train")
bound_dir = os.path.join(BASE_DIR, "compressor_bound")
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)


def parse_power_series(s):
    if pd.isna(s) or s == "":
        return np.array([])

    return np.array([
        float(x)
        for x in str(s).replace(" ", "").split(",")
    ])


def get_bounds_from_json(temp, TEMP_BOUNDS):
    key = str(round(float(temp), 1))

    if key not in TEMP_BOUNDS:
        return np.nan, np.nan

    return (
        float(TEMP_BOUNDS[key][0]),
        float(TEMP_BOUNDS[key][1])
    )


def extract_power_features(row, TEMP_BOUNDS):

    power = parse_power_series(row["power"])
    temp = row["temperature"]

    if len(power) == 0:
        return pd.Series({})

    bmin, bmax = get_bounds_from_json(temp, TEMP_BOUNDS)

    if np.isnan(bmin) or np.isnan(bmax):
        return pd.Series({})

    if bmax <= bmin:
        return pd.Series({})

    width = (bmax - bmin) + 1e-6


    norm = (power - bmin) / width

    feats = {}


    inside_mask = (norm >= 0) & (norm <= 1)
    tail = norm[-1:]

    inside_tail = (tail >= 0) & (tail <= 1)

    feats["final_inside_frac"] = np.mean(inside_tail)
    feats["inside_frac"] = np.mean(inside_mask)


    feats["max_over"] = max(norm.max() - 1, 0)
    over = np.maximum(power - bmax, 0)
    under = np.maximum(bmin - power, 0)

    diff = np.diff(power)

    local_jump = np.max(np.abs(diff))
    baseline = np.median(np.abs(diff)) + 1e-6

    feats["has_spike"] = int(local_jump / baseline > 5)

    feats["max_under"] = max(0 - norm.min(), 0)

    return pd.Series(feats)

FEATURE_ORDER = [
    "final_inside_frac",
    "inside_frac",
    "max_over",
    "max_under",
    "has_spike",
]


for filename in os.listdir(train_dir):

    if not filename.endswith("_train.csv"):
        continue

    model_name = filename.replace("_train.csv", "")

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    train_path = os.path.join(train_dir, filename)

    bound_path = os.path.join(
        bound_dir,
        f"{model_name}_boundaries.json"
    )

    if not os.path.exists(bound_path):
        print("NO BOUND FILE:", bound_path)
        continue


    with open(bound_path, "r", encoding="utf-8") as f:
        TEMP_BOUNDS = json.load(f)


    df = pd.read_csv(train_path)

    print("ROWS:", len(df))
    print(df["label"].value_counts())

    X_full = df.apply(
        lambda r: extract_power_features(r, TEMP_BOUNDS),
        axis=1
    )

    X = X_full.reindex(columns=FEATURE_ORDER).fillna(0.0)

    y = df["label"].values

    print("\nFEATURE SAMPLE:")
    print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,

        subsample=0.7,
        colsample_bytree=0.7,

        reg_lambda=3.0,
        reg_alpha=0.5,

        min_child_weight=5,
        gamma=0.3,

        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)

    print("\nROC AUC:", round(auc, 6))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, digits=4))


    print("\nFEATURE IMPORTANCE:")

    importance = model.feature_importances_

    for feat, score in sorted(
        zip(FEATURE_ORDER, importance),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feat:20s} {score:.6f}")

    booster = model.get_booster()

    booster.set_attr(
        boundary_table=json.dumps(TEMP_BOUNDS)
    )

    model_path = os.path.join(
        model_dir,
        f"{model_name}_model.json"
    )

    booster.save_model(model_path)

    print("\nMODEL SAVED:")
    print(model_path)