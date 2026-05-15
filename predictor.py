import pandas as pd
import numpy as np
import json
import xgboost as xgb
from xgboost import XGBClassifier

MODEL_PATH = "models/SZ65_0551_model.json"

LOW_THR = 0.20
HIGH_THR = 0.80

FEATURE_ORDER = [
    "final_inside_frac",
    "inside_frac",
    "max_over",
    "max_under",
    "has_spike",
]

model = XGBClassifier()
model.load_model(MODEL_PATH)

booster = model.get_booster()

boundary_json = booster.attr("boundary_table")
BOUNDARY_TABLE = json.loads(boundary_json) if boundary_json else {}

print("BOUNDARY TABLE SIZE:", len(BOUNDARY_TABLE))


def parse_power_series(s):
    if pd.isna(s) or s == "":
        return np.array([])
    return np.array([float(x) for x in str(s).replace(" ", "").split(",")])


def get_bounds_from_model(temp):
    key = str(round(float(temp), 1))

    if key not in BOUNDARY_TABLE:
        return None, None

    bmin, bmax = BOUNDARY_TABLE[key]
    return float(bmin), float(bmax)


def extract_power_features(row):

    power = parse_power_series(row["power"])
    if len(power) == 0:
        return pd.Series({})

    temp = row["temperature"]
    bmin, bmax = get_bounds_from_model(temp)

    if bmin is None or bmax is None or bmax <= bmin:
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
    feats["max_under"] = max(0 - norm.min(), 0)
    over = np.maximum(power - bmax, 0)
    under = np.maximum(bmin - power, 0)

    diff = np.diff(power)

    local_jump = np.max(np.abs(diff))
    baseline = np.median(np.abs(diff)) + 1e-6

    feats["has_spike"] = int(local_jump / baseline > 5) 

    return pd.Series(feats)

def explain_row(i, X_test, contribs):

    row = contribs[i]

    feature_contribs = row[:-1]
    bias = row[-1]

    raw_score = bias + np.sum(feature_contribs)
    prob = 1 / (1 + np.exp(-raw_score))

    print("\n--- EXPLANATION ---")
    print("BIAS:", bias)
    print("RAW SCORE:", raw_score)
    print("RECONSTRUCTED PROB:", prob)

    print("\nFEATURE IMPACT:")

    sorted_idx = np.argsort(np.abs(feature_contribs))[::-1]

    for j in sorted_idx:
        name = FEATURE_ORDER[j]
        val = feature_contribs[j]
        sign = "+" if val >= 0 else "-"

        print(f"{name:15s} {sign}{abs(val):.4f}")

df = pd.read_csv("test.csv")

features = df.apply(extract_power_features, axis=1)

X_test = features.reindex(columns=FEATURE_ORDER).fillna(0.0)

print("\nX shape:", X_test.shape)


y_prob = model.predict_proba(X_test)[:, 1]

y_pred = np.full(len(y_prob), 3, dtype=int)
y_pred[y_prob <= LOW_THR] = 2
y_pred[y_prob >= HIGH_THR] = 1


dmat = xgb.DMatrix(X_test)
contribs = booster.predict(dmat, pred_contribs=True)


for i in range(len(df)):

    print("\n================ ROW", i, "================")

    print("BARCODE:", df.loc[i, "barcode"])
    print("TEMP:", df.loc[i, "temperature"])

    print("\n--- FEATURES ---")
    print(X_test.iloc[i])

    print("\nPROB:", y_prob[i])
    print("PRED:", y_pred[i])

    explain_row(i, X_test, contribs)


print("\n=========== SUMMARY ===========")

for i in range(len(df)):
    print(
        df.loc[i, "barcode"],
        "| prob=", round(y_prob[i], 4),
        "| pred=", y_pred[i]
    )

print("\nMODEL FEATURES:", model.get_booster().feature_names)
print("INPUT FEATURES:", X_test.columns.tolist())

print("\nCONTRIB SHAPE:", contribs.shape)