import io, re, time, datetime
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

CDL_STAT_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat"

# ---- CONFIG ----
FIPS = "29205"   # Shelby County, MO
this_year = datetime.date.today().year
PAST_YEARS_TO_PULL = [this_year-3, this_year-2, this_year-1]   # last 3 released years
FUTURE_YEARS = [this_year, this_year+1, this_year+2]           # next 3 planning years
OUT_PATH = Path("data_work/county_year_features.csv")

def _download(url: str, timeout=60) -> str:
    r = requests.get(url, timeout=timeout); r.raise_for_status(); return r.text

def fetch_year(fips: str, year: int, retries=3, sleep_s=1.5) -> pd.DataFrame:
    """
    Fetch CDL category/area for a past year and normalize to columns: ['category','acres'].
    Handles schema variants: 'Acres', 'Area (acres)', 'COUNT' pixels, 'Class_Name', etc.
    """
    last_err = None
    for _ in range(retries):
        try:
            # Call the service
            resp = requests.post(
                CDL_STAT_URL,
                data={"year": str(year), "fips": str(fips), "format": "csv"},
                timeout=30,
            )
            resp.raise_for_status()
            txt = resp.text.strip()

            # Sometimes the CSV is inline, sometimes it's an XML with a return URL
            if "category" in txt.lower() and "class" not in txt.lower() and "<" not in txt[:200]:
                df = pd.read_csv(io.StringIO(txt))
            else:
                m = re.search(r"<returnurl>(https?://[^<]+\.csv)</returnurl>", txt, flags=re.I)
                if not m:
                    m = re.search(r"returnurl[^>]*>(https?://[^<]+\.csv)<", txt, flags=re.I)
                if not m:
                    raise ValueError(f"Unexpected response for {year}: {txt[:200]}...")
                df = pd.read_csv(io.StringIO(_download(m.group(1))))

            # Normalize headers
            original_cols = df.columns.tolist()
            cols_map = {c: c.strip() for c in df.columns}
            df.columns = (
                df.columns.str.strip()
                         .str.lower()
                         .str.replace(r"\s+", "_", regex=True)
                         .str.replace(r"[()]", "", regex=True)
            )

            # Candidate columns
            # category-like
            cat_cands = [c for c in df.columns if c in ("category", "class", "class_name", "classname", "classvalue", "class_value")]
            if not cat_cands:
                # sometimes it's 'crop', 'name'
                cat_cands = [c for c in df.columns if any(k in c for k in ("crop","name","label","class"))]
            if not cat_cands:
                raise ValueError(f"No category/class column found. Columns: {original_cols}")
            cat_col = cat_cands[0]

            # acres-like
            acre_cands = [c for c in df.columns if c in ("acres", "area_acres", "area_acre", "area_acres_est")]
            if not acre_cands:
                # look for "area_acres" variations like "area_acres" after normalization from "Area (acres)"
                acre_cands = [c for c in df.columns if "acre" in c and "per" not in c]
            acres_series = None

            if acre_cands:
                acres_series = pd.to_numeric(df[acre_cands[0]], errors="coerce")
            else:
                # Fallback: compute acres from pixel COUNT if present (CDL pixels are 30m x 30m)
                # 1 pixel ≈ 0.222394 acres
                count_cands = [c for c in df.columns if c in ("count", "pixels", "pixel_count")]
                if count_cands:
                    acres_series = pd.to_numeric(df[count_cands[0]], errors="coerce") * 0.222394
                else:
                    # Some tables have 'area' with units undocumented; try it as-is
                    area_cands = [c for c in df.columns if c.startswith("area")]
                    if area_cands:
                        acres_series = pd.to_numeric(df[area_cands[0]], errors="coerce")
                    else:
                        raise ValueError(f"No acres/area/count columns found. Columns: {original_cols}")

            out = pd.DataFrame({
                "category": df[cat_col].astype(str),
                "acres": acres_series.fillna(0.0)
            })
            return out
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"CDL fetch failed for year {year}, fips {fips}: {last_err}")

def compute_shares(df: pd.DataFrame) -> dict:
    t = df.copy()
    t["category"] = t["category"].astype(str)
    t["acres"] = pd.to_numeric(t["acres"], errors="coerce").fillna(0.0)
    denom = t["acres"].sum()
    if denom <= 0:
        return {"crop_share_corn": 0.0, "crop_share_soy": 0.0, "crop_diversity_index": 0.0}
    corn = t.loc[t["category"].str.lower().eq("corn"), "acres"].sum()
    soy  = t.loc[t["category"].str.lower().isin(["soybeans","soybean"]), "acres"].sum()
    p_corn = float(corn/denom); p_soy = float(soy/denom)
    shares = (t["acres"]/denom).clip(lower=1e-12)
    shannon = -float((shares * np.log(shares)).sum())
    return {"crop_share_corn": round(p_corn,6), "crop_share_soy": round(p_soy,6), "crop_diversity_index": round(shannon,6)}

def upsert_row(base: pd.DataFrame, fips: str, year: int, shares: dict) -> pd.DataFrame:
    row = {
        "county_fips": str(fips),
        "year": int(year),
        **shares,
        "usgs_kgai_total": pd.NA,
        "usgs_kgai_glyphos": pd.NA,
        "usgs_kgai_glufos": pd.NA,
        "gdd_apr_sep": pd.NA,
        "precip_apr_sep": pd.NA,
        "extreme_heat_days": pd.NA,
        "soil_om_mean": pd.NA,
        "soil_texture_class": "silt_loam",
        "soil_ph_mean": pd.NA,
        "weed_occurrence_prior": 1,
    }
    if base is None:
        return pd.DataFrame([row])
    mask = (base["county_fips"].astype(str)==str(fips)) & (base["year"].astype(int)==int(year))
    if mask.any():
        base.loc[base.index[mask][0], :] = {**base.loc[base.index[mask][0], :].to_dict(), **row}
    else:
        base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    return base

def main():
    base = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else None

    pulled = []
    print(f"Fetching past years (must be < {this_year}): {PAST_YEARS_TO_PULL}")
    for y in PAST_YEARS_TO_PULL:
        if y >= this_year:
            print(f"Skip {y} (future)"); continue
        try:
            df = fetch_year(FIPS, y)
            s = compute_shares(df)
            pulled.append(s)
            base = upsert_row(base, FIPS, y, s)
            print(f"✓ Past {y}: {s}")
        except Exception as e:
            print(f"⚠ Skipping {y}: {e}")

    if not pulled:
        raise SystemExit("No past years fetched; cannot forecast future years.")

    # Forecast = average of pulled shares (fallback to last if only 1)
    if len(pulled) >= 2:
        avg = {k: float(np.mean([s[k] for s in pulled])) for k in pulled[0].keys()}
    else:
        avg = pulled[-1]

    print(f"Writing forecast for future years: {FUTURE_YEARS} → {avg}")
    for y in FUTURE_YEARS:
        base = upsert_row(base, FIPS, y, avg)

    base = base.sort_values(["county_fips","year"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT_PATH, index=False)
    print(f"Updated {OUT_PATH} with {len(pulled)} past + {len(FUTURE_YEARS)} future years.")

if __name__ == "__main__":
    main()