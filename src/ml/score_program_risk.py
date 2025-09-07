import numpy as np
import pandas as pd, joblib

features = pd.read_csv("data_work/county_year_features.csv")
programs = pd.read_csv("data_work/programs.csv")
model = joblib.load("models/risk_model_xgb_cal.joblib")

# Build feature matrix
drop_cols = {"county_fips","year","soil_texture_class"}
FEATS = [c for c in features.columns if c not in drop_cols]
X = features[FEATS] if FEATS else pd.DataFrame(index=features.index)

# Get probability of class "1" (resistance). Handle single-class models safely.
p1 = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
    # If model knows both classes, pick the column for class==1
    if hasattr(model, "classes_") and 1 in getattr(model, "classes_", []):
        pos_idx = list(model.classes_).index(1)
        p1 = proba[:, pos_idx]
    else:
        # Model never saw class 1 → use a small prior
        p1 = np.full(len(X), 0.05, dtype=float)
else:
    # Fallback: use predictions as probabilities (coarse)
    preds = model.predict(X)
    p1 = preds.astype(float)

features = features.copy()
features["risk_next"] = p1

# Expand to program-year table
rows = []
for _, r in features.iterrows():
    for _, p in programs.iterrows():
        rows.append({
            "county_fips": r["county_fips"],
            "year": int(r["year"]),
            "program_id": p["program_id"],
            "R_t_m": float(r["risk_next"]),
            "cost_per_acre": float(p["cost_per_acre"]),
        })

out = pd.DataFrame(rows)
out.to_csv("data_work/risk_by_program.csv", index=False)
print("Wrote data_work/risk_by_program.csv with", len(out), "rows")
