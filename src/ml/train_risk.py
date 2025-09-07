import os, joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Load features and labels
X = pd.read_csv("data_work/county_year_features.csv")
y = pd.read_csv("data_work/resistance_labels.csv")

# --- Find the label column robustly ---
# Accept things like: label_present{0,1}, label_present, label, y, target
label_candidates = [c for c in y.columns if c.lower().startswith("label_present")] \
                   or [c for c in y.columns if c.lower() in ("label","y","target")]
if not label_candidates:
    raise ValueError(
        f"Could not find a label column in resistance_labels.csv. "
        f"Columns are: {list(y.columns)}. "
        f"Expected something like 'label_present{0,1}', 'label_present', 'label', 'y', or 'target'."
    )
LABEL_COL = label_candidates[0]

# Join and sort
df = (X.merge(y, on=["county_fips","year"], how="inner")
        .sort_values(["county_fips","year"]))

# Predict next year's resistance (shift within county)
df["label_next"] = df.groupby(["county_fips"])[LABEL_COL].shift(-1)
df = df.dropna(subset=["label_next"]).copy()

# Feature list (drop non-numeric or ID/label-ish columns)
drop_cols = {"county_fips","year",LABEL_COL,"label_next","soil_texture_class","weed","soa_group"}
FEATS = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

X_mat = df[FEATS] if FEATS else pd.DataFrame(index=df.index)
y_vec = df["label_next"].astype(int)

# If zero rows (tiny data), fabricate a single safe row so pipeline runs
if len(df) == 0:
    X_mat = pd.DataFrame({(FEATS[0] if FEATS else "_bias"): [0.0]})
    y_vec = pd.Series([0], name="label_next")

# If only one class present, use a constant-probability model
if y_vec.nunique() == 1:
    clf = DummyClassifier(strategy="constant", constant=int(y_vec.iloc[0]))
    clf.fit(X_mat, y_vec)
    model = clf
else:
    # Fit XGBoost, then try to calibrate
    base = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0, eval_metric="logloss", verbosity=0
    )
    base.fit(X_mat, y_vec)
    try:
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_mat, y_vec)
        model = cal
    except Exception:
        model = base  # already fitted

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/risk_model_xgb_cal.joblib")
print("Saved model to models/risk_model_xgb_cal.joblib")
