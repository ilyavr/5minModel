
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from xgboost import XGBClassifier

# ---------------- CONFIG ----------------

MODEL_PATH = "models/SZ59_0412_model.json"

LOW_THR = 0.20
HIGH_THR = 0.80

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
        print(f"[WARN] No bounds for temp={temp} (key={key})")
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
        print(f"[WARN] Invalid bounds → skip row | temp={temp}")
        return pd.Series({})

    n = len(power)
    width = (bmax - bmin) + 1e-6

    p_mean = power.mean()
    p_std = power.std()
    p_min = power.min()
    p_max = power.max()

    # --- отклонения (НОРМАЛИЗОВАННЫЕ) ---
    under = np.maximum(bmin - power, 0) / width
    over  = np.maximum(power - bmax, 0) / width

    feats = {}

    feats["under_mean"] = under.mean()
    feats["under_max"]  = under.max()

    feats["over_mean"]  = over.mean()
    feats["over_max"]   = over.max()

    # --- частота ---
    feats["under_frac"] = np.mean(power < bmin)
    feats["over_frac"]  = np.mean(power > bmax)

    # --- положение ---
    feats["mean_pos"] = (p_mean - bmin) / width

    # --- запас ---
    feats["margin_to_min"] = (p_min - bmin) / width
    feats["margin_to_max"] = (bmax - p_max) / width

    # --- динамика ---
    feats["max_jump"] = np.max(np.abs(np.diff(power))) / width if n > 1 else 0.0

    if n > 1:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, power, 1)
        trend = slope * x + intercept
        residual = np.mean(np.abs(power - trend)) / width
    else:
        slope = 0.0
        residual = 0.0

    feats["trend_slope"] = slope / width

    # --- статистика ---
    feats["power_cv"] = p_std / (p_mean + 1e-6)

    # debug
    feats["_debug_bmin"] = bmin
    feats["_debug_bmax"] = bmax

    return pd.Series(feats)


# ---------------- LOAD DATA ----------------

df = pd.read_csv("test.csv")

power_features = df.apply(extract_power_features, axis=1)

debug_cols = [c for c in power_features.columns if c.startswith("_debug")]
debug_df = power_features[debug_cols]

X_test = power_features.drop(columns=debug_cols, errors="ignore")
print(X_test.dtypes)
X_test = X_test.reindex(columns=FEATURE_ORDER).fillna(0.0)
print(X_test.dtypes)

print(X_test.head())
print(X_test.nunique())

# ---------------- PREDICT ----------------

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = np.full(len(y_prob), 3, dtype=int)
y_pred[y_prob <= LOW_THR] = 2
y_pred[y_prob >= HIGH_THR] = 1

# ---------------- SHAP ----------------

dmat = xgb.DMatrix(X_test)
contribs = booster.predict(dmat, pred_contribs=True)
print(dmat.num_row(), dmat.num_col())
# ---------------- OUTPUT ----------------

for i in range(len(X_test)):
    print(f"\n================ ROW {i+1} ================")

    print("BARCODE:", df.loc[i, "barcode"])
    print("TEMP:", df.loc[i, "temperature"])

    print("BOUNDS:",
          debug_df.iloc[i]["_debug_bmin"],
          debug_df.iloc[i]["_debug_bmax"])

    print("\n--- REAL FEATURES ---")
    print(X_test.iloc[i])

    print("\n--- PREDICTION ---")
    print("PROB:", y_prob[i])
    print("PRED:", y_pred[i])

    row_contribs = contribs[i]
    feature_contribs = row_contribs[:-1]
    bias = row_contribs[-1]

    sorted_idx = np.argsort(np.abs(feature_contribs))[::-1]

    print("\n--- TOP FEATURE CONTRIBUTIONS ---")
    for j in sorted_idx[:10]:
        name = FEATURE_ORDER[j]
        print(f"{name}: {feature_contribs[j]:.4f}")

    print("bias:", bias)

# ---------------- SUMMARY ----------------

print("\n=========== SUMMARY ===========")
for i in range(len(df)):
    print(f"{df.loc[i,'barcode']} | prob={y_prob[i]:.4f} | pred={y_pred[i]}")

print("\nMODEL FEATURES:", model.get_booster().feature_names)
print("INPUT FEATURES:", X_test.columns.tolist())

print(contribs.shape)
pred_leaf = booster.predict(dmat, pred_leaf=True)

for i in range(len(pred_leaf)):
    print(i, pred_leaf[i][:10])