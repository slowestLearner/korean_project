import os
os.environ["OMP_NUM_THREADS"] = "1"

import re
import glob
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# ============================================================
# LOAD ENV
# ============================================================
load_dotenv("../.env")

DATA_DIR = Path(os.getenv("DATA_DIR"))
CODE_DIR = Path(os.getenv("CODE_DIR"))

# ============================================================
# USER SETTINGS
# ============================================================
TRADES_DIR = DATA_DIR / "data/investors/inst_monthly"
CHAR_FILE  = DATA_DIR / "data/jkp/jkp_full.parquet"
OUT_DIR    = DATA_DIR / "data/investors/lasso_out_monthly"

CLUSTER_MAP_CSV = DATA_DIR / "data/jkp/jkp_cluster_map.csv"
SIGN_CSV        = DATA_DIR / "data/jkp/sign.csv"

START_YEAR = 2010
END_YEAR   = 2023

TARGET_COL_RAW = "trade_std"
Y_COL = "y_std_jm"

MIN_OBS_PER_INV = 200

# ---- Missing values handling for X (theme_l1) ----
IMPUTE_MODE = "drop_any_missing"   # "zero_only" | "drop_any_missing" | "zero_plus_indicators"

# ---- LASSO ----
STANDARDIZE_X_WITHIN_INV = True
FIT_INTERCEPT = True

USE_AUTO_ALPHA = True
FIXED_ALPHA = 0.10

ALPHA_GRID = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
TARGET_MEAN_NNZ_SIGNAL = 4
N_SAMPLE_INV_FOR_ALPHA = 250

# ---- Classification filters ----
MIN_R2_FOR_CLASS = 0.02
MIN_ABS_TOPB_FOR_CLASS = 0.06
MAX_NNZ_SIGNAL_FOR_CLASS = 12

USE_STRENGTH_FILTER = True
MIN_STRENGTH_FOR_CLASS = 0.02
SIGNED_GROUPS = True

# Parallel
N_JOBS = -1
RANDOM_SEED = 123

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def stable_seed(*parts, base=RANDOM_SEED) -> int:
    s = "|".join(map(str, parts)) + f"|base={base}"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % (2**32 - 1))

def trade_files_for_year(year: int):
    """
    Cluster monthly files look like:
      inst_trades_monthly_201001.parquet
      inst_trades_monthly_201002.parquet
      ...
    """
    pat1 = str(TRADES_DIR / f"inst_trades_monthly_{year}*.parquet")
    pat2 = str(TRADES_DIR / f"inst_trade_monthly_{year}*.parquet")
    paths = sorted(set(glob.glob(pat1) + glob.glob(pat2)))
    return paths

def jm_standardize_y(df: pd.DataFrame, raw_y: str, out_y: str) -> pd.DataFrame:
    g = df.groupby(["encryp_acnt_no", "year", "month"], sort=False)[raw_y]
    mu = g.transform("mean")
    sd = g.transform("std")
    df[out_y] = (df[raw_y] - mu) / sd
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[out_y]).copy()
    return df

def build_lasso_pipeline(alpha: float, inv: str):
    steps = []
    if STANDARDIZE_X_WITHIN_INV:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append((
        "lasso",
        Lasso(
            alpha=float(alpha),
            fit_intercept=FIT_INTERCEPT,
            max_iter=10000,
            tol=1e-4,
            random_state=stable_seed("LASSO", inv, alpha),
            selection="cyclic",
        )
    ))
    return Pipeline(steps)

def fit_one_investor(inv: str, df_inv: pd.DataFrame, feature_cols_all, feature_cols_signal, alpha: float):
    df_inv = df_inv.dropna(subset=[Y_COL] + feature_cols_all)
    n = df_inv.shape[0]
    if n < MIN_OBS_PER_INV:
        return None

    X = df_inv[feature_cols_all].to_numpy(dtype=np.float64)
    y = df_inv[Y_COL].to_numpy(dtype=np.float64)

    model = build_lasso_pipeline(alpha, inv)
    model.fit(X, y)

    lasso = model.named_steps["lasso"]
    coef = lasso.coef_.astype(np.float32)
    intercept = float(lasso.intercept_) if FIT_INTERCEPT else 0.0

    yhat = model.predict(X)
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ssr / sst) if sst > 0 else np.nan

    nnz_total = int(np.sum(np.abs(coef) > 1e-8))

    idx_signal = [feature_cols_all.index(c) for c in feature_cols_signal]
    coef_signal = coef[idx_signal]
    nnz_signal = int(np.sum(np.abs(coef_signal) > 1e-8))

    row = {
        "encryp_acnt_no": inv,
        "n_obs": int(n),
        "intercept": intercept,
        "alpha": float(alpha),
        "r2_in": r2,
        "nnz_total": nnz_total,
        "nnz_signal": nnz_signal,
    }
    for c, v in zip(feature_cols_all, coef):
        row[f"b_{c}"] = float(v)
    return row

def pick_alpha_for_year(merged: pd.DataFrame, feature_cols_all, feature_cols_signal):
    counts = merged["encryp_acnt_no"].value_counts()
    invs = counts[counts >= MIN_OBS_PER_INV].index.tolist()
    if not invs:
        return None

    rng = np.random.default_rng(stable_seed("ALPHA_PICK"))
    invs_sample = rng.choice(
        invs,
        size=min(len(invs), N_SAMPLE_INV_FOR_ALPHA),
        replace=False
    ).tolist()

    grouped = {inv: merged[merged["encryp_acnt_no"] == inv] for inv in invs_sample}

    results = []
    for a in ALPHA_GRID:
        nnz_list = []
        for inv, df_inv in grouped.items():
            r = fit_one_investor(inv, df_inv, feature_cols_all, feature_cols_signal, alpha=a)
            if r is not None:
                nnz_list.append(r["nnz_signal"])
        mean_nnz = float(np.mean(nnz_list)) if nnz_list else np.inf
        results.append((a, mean_nnz))

    best = min(results, key=lambda t: abs(t[1] - TARGET_MEAN_NNZ_SIGNAL))
    return best[0], results

# ============================================================
# THEME BUILDING
# ============================================================
def sanitize_theme_name(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]", "_", str(s).strip())
    s2 = re.sub(r"_+", "_", s2).strip("_")
    if s2 == "":
        s2 = "Theme"
    return s2

def load_maps():
    if not CLUSTER_MAP_CSV.exists():
        raise FileNotFoundError(f"Missing cluster map: {CLUSTER_MAP_CSV}")
    if not SIGN_CSV.exists():
        raise FileNotFoundError(f"Missing sign file: {SIGN_CSV}")

    cmap = pd.read_csv(CLUSTER_MAP_CSV)
    sgn = pd.read_csv(SIGN_CSV)

    if not {"char", "cluster"}.issubset(cmap.columns):
        raise ValueError("jkp_cluster_map.csv must have columns: char, cluster")
    if not {"char", "direction"}.issubset(sgn.columns):
        raise ValueError("sign.csv must have columns: char, direction")

    cmap["char"] = cmap["char"].astype(str)
    cmap["cluster"] = cmap["cluster"].astype(str)

    sgn["char"] = sgn["char"].astype(str)
    sgn["direction"] = pd.to_numeric(sgn["direction"], errors="coerce")
    if sgn["direction"].isna().any():
        bad = sgn.loc[sgn["direction"].isna(), "char"].head(10).tolist()
        raise ValueError(f"sign.csv has non-numeric direction for: {bad}")

    char_to_theme = dict(zip(cmap["char"], cmap["cluster"]))
    char_to_dir = dict(zip(sgn["char"], sgn["direction"].astype(int)))
    return char_to_theme, char_to_dir

def build_jkp_theme_file():
    print("Loading jkp_full.parquet ...")
    df = pd.read_parquet(CHAR_FILE)

    for k in ["isin", "year", "month"]:
        if k not in df.columns:
            raise RuntimeError(f"Expected key column '{k}' in {CHAR_FILE}")

    df["isin"] = df["isin"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)

    char_to_theme, char_to_dir = load_maps()

    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    base_chars = set()
    for c in rank_cols:
        b = c.replace("rank_", "")
        if b.endswith("_l1"):
            b = b[:-3]
        base_chars.add(b)

    usable_chars = sorted([ch for ch in base_chars if ch in char_to_theme and ch in char_to_dir])
    if not usable_chars:
        raise RuntimeError("No usable chars found that match both jkp_cluster_map.csv and sign.csv.")

    themes = sorted({char_to_theme[ch] for ch in usable_chars})
    theme_name_map = {t: sanitize_theme_name(t) for t in themes}

    print(f"  usable base chars: {len(usable_chars)}")
    print(f"  themes detected: {len(themes)}")

    out = df[["isin", "year", "month"]].copy()

    theme_to_cols_cur = {t: [] for t in themes}
    theme_to_cols_l1 = {t: [] for t in themes}
    signed_cols = {}

    for ch in usable_chars:
        t = char_to_theme[ch]
        d = float(char_to_dir[ch])

        c_cur = f"rank_{ch}"
        c_l1 = f"rank_{ch}_l1"

        if c_cur in df.columns:
            signed_name = f"__signed__{c_cur}"
            signed_cols[signed_name] = df[c_cur] * d
            theme_to_cols_cur[t].append(signed_name)

        if c_l1 in df.columns:
            signed_name = f"__signed__{c_l1}"
            signed_cols[signed_name] = df[c_l1] * d
            theme_to_cols_l1[t].append(signed_name)

    if signed_cols:
        out = pd.concat([out, pd.DataFrame(signed_cols, index=df.index)], axis=1)

    for t in themes:
        t2 = theme_name_map[t]
        cur_cols = theme_to_cols_cur[t]
        l1_cols = theme_to_cols_l1[t]

        if cur_cols:
            out[f"rank_{t2}"] = out[cur_cols].mean(axis=1, skipna=True).astype(np.float32)
        if l1_cols:
            out[f"rank_{t2}_l1"] = out[l1_cols].mean(axis=1, skipna=True).astype(np.float32)

    tmp_cols = list(signed_cols.keys())
    if tmp_cols:
        out = out.drop(columns=tmp_cols)

    out = out.copy()

    theme_path = OUT_DIR / "jkp_full_theme.parquet"
    out.to_parquet(theme_path, index=False)
    print(f"Saved theme panel: {theme_path}  rows={out.shape[0]:,}  cols={out.shape[1]:,}")

    return theme_path

# ============================================================
# CLASSIFICATION (POSITIVE BETAS ONLY)
# ============================================================
def classify_investors_positive_only(lasso_df: pd.DataFrame, feature_cols_all, feature_cols_signal):
    beta_cols_signal = [f"b_{c}" for c in feature_cols_signal]
    for bc in beta_cols_signal:
        if bc not in lasso_df.columns:
            lasso_df[bc] = 0.0

    Bsig = lasso_df[beta_cols_signal].to_numpy(dtype=np.float64)

    Bpos = np.where(Bsig > 0, Bsig, 0.0)
    has_pos = (Bpos.max(axis=1) > 0)

    top_idx = Bpos.argmax(axis=1)
    top_beta = Bsig[np.arange(Bsig.shape[0]), top_idx]
    top_abs = np.abs(top_beta)
    top_feat = np.array(beta_cols_signal, dtype=object)[top_idx]

    out = lasso_df[["year", "encryp_acnt_no", "n_obs", "alpha", "r2_in", "nnz_total", "nnz_signal"]].copy()
    out["top_feature"] = top_feat
    out["top_beta"] = top_beta.astype(np.float32)
    out["top_abs_beta"] = top_abs.astype(np.float32)

    out["strength"] = (
        out["top_abs_beta"] * np.sqrt(np.maximum(out["r2_in"].fillna(0).to_numpy(), 0))
    ).astype(np.float32)

    base_ok = (
        (out["r2_in"].fillna(0) >= MIN_R2_FOR_CLASS)
        & (out["top_abs_beta"] >= MIN_ABS_TOPB_FOR_CLASS)
        & (out["nnz_signal"] <= MAX_NNZ_SIGNAL_FOR_CLASS)
        & has_pos
    )
    ok = base_ok & (out["strength"] >= MIN_STRENGTH_FOR_CLASS) if USE_STRENGTH_FILTER else base_ok
    out["is_classified"] = ok.astype(int)

    base_group = out["top_feature"].str.replace("^b_", "", regex=True)
    sign = np.where(out["top_beta"] >= 0, "+", "-")

    if SIGNED_GROUPS:
        out["group"] = np.where(out["is_classified"] == 1, base_group + ":" + sign, "")
        out["direction"] = np.where(out["is_classified"] == 1, sign, "")
    else:
        out["group"] = np.where(out["is_classified"] == 1, base_group, "")
        out["direction"] = ""

    return out

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()

    theme_file = build_jkp_theme_file()

    print("Loading jkp_full_theme.parquet (only rank_*_l1 theme features)...")
    chars_all = pd.read_parquet(theme_file)
    chars_all["year"] = pd.to_numeric(chars_all["year"], errors="raise").astype(int)
    chars_all["month"] = pd.to_numeric(chars_all["month"], errors="raise").astype(int)
    chars_all["isin"] = chars_all["isin"].astype(str)

    feature_cols_signal = [c for c in chars_all.columns if c.startswith("rank_") and c.endswith("_l1")]
    if len(feature_cols_signal) == 0:
        raise RuntimeError("No theme lag features found. Expected columns like rank_<Theme>_l1.")

    chars = chars_all[["isin", "year", "month"] + feature_cols_signal].copy()
    del chars_all
    print(f"  theme lag signals: {len(feature_cols_signal)}  rows: {chars.shape[0]:,}")

    for year in range(START_YEAR, END_YEAR + 1):
        trade_paths = trade_files_for_year(year)
        if not trade_paths:
            print(f"[{year}] SKIP: no trade files found.")
            continue

        print(f"\n[{year}] Loading trades ({len(trade_paths)} files)...")

        must = ["year", "month", "isin", "encryp_acnt_no", TARGET_COL_RAW]
        extra = ["trade", "trdvol", "num"]

        chunks = []
        for p in trade_paths:
            df = pd.read_parquet(p)
            keep = [c for c in (must + extra) if c in df.columns]
            df = df[keep].copy()
            df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
            df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
            df["isin"] = df["isin"].astype(str)
            df["encryp_acnt_no"] = df["encryp_acnt_no"].astype(str)
            chunks.append(df)

        trades = pd.concat(chunks, ignore_index=True)
        print(f"[{year}] trades rows: {trades.shape[0]:,}")

        merged = trades.merge(chars, on=["isin", "year", "month"], how="left", validate="m:1")

        merged = merged.dropna(subset=[TARGET_COL_RAW]).copy()
        if merged.empty:
            print(f"[{year}] SKIP: no rows with non-missing y.")
            continue

        merged = jm_standardize_y(merged, raw_y=TARGET_COL_RAW, out_y=Y_COL)
        print(f"[{year}] rows after investor-month y standardization: {merged.shape[0]:,}")
        if merged.empty:
            print(f"[{year}] SKIP: empty after y standardization.")
            continue

        if IMPUTE_MODE == "drop_any_missing":
            merged = merged.dropna(subset=feature_cols_signal).copy()
            feature_cols_all = feature_cols_signal[:]
        elif IMPUTE_MODE == "zero_only":
            merged[feature_cols_signal] = merged[feature_cols_signal].fillna(0.0)
            feature_cols_all = feature_cols_signal[:]
        elif IMPUTE_MODE == "zero_plus_indicators":
            miss_indicators = {}
            for c in feature_cols_signal:
                miss_indicators[f"miss_{c}"] = merged[c].isna().astype(np.float32)
            indicator_df = pd.DataFrame(miss_indicators, index=merged.index)
            merged = pd.concat([merged, indicator_df], axis=1)
            merged[feature_cols_signal] = merged[feature_cols_signal].fillna(0.0)
            feature_cols_all = feature_cols_signal[:] + list(indicator_df.columns)
        else:
            raise ValueError(f"Unknown IMPUTE_MODE={IMPUTE_MODE}")

        counts = merged["encryp_acnt_no"].value_counts()
        keep_invs = counts[counts >= MIN_OBS_PER_INV].index
        merged = merged[merged["encryp_acnt_no"].isin(keep_invs)].copy()
        print(f"[{year}] investors kept (>= {MIN_OBS_PER_INV} obs): {len(keep_invs):,}")
        print(f"[{year}] rows after investor filter: {merged.shape[0]:,}")
        if merged.empty:
            print(f"[{year}] SKIP: no investors meet MIN_OBS_PER_INV.")
            continue

        if USE_AUTO_ALPHA:
            picked = pick_alpha_for_year(merged, feature_cols_all, feature_cols_signal)
            if picked is None:
                print(f"[{year}] SKIP: alpha picking failed.")
                continue
            alpha_year, grid_res = picked
            grid_str = ", ".join([f"{a}:{m:.1f}" for a, m in grid_res])
            print(f"[{year}] auto alpha picked={alpha_year} (alpha:mean_nnz_signal => {grid_str})")
        else:
            alpha_year = float(FIXED_ALPHA)
            print(f"[{year}] fixed alpha={alpha_year}")

        grouped = list(merged.groupby("encryp_acnt_no", sort=False))
        print(f"[{year}] fitting LASSO per investor... (features_all={len(feature_cols_all)} signals={len(feature_cols_signal)})")

        res = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(fit_one_investor)(inv, df_inv, feature_cols_all, feature_cols_signal, alpha_year)
            for inv, df_inv in grouped
        )
        res = [r for r in res if r is not None]
        if not res:
            print(f"[{year}] SKIP: no successful fits.")
            continue

        lasso_df = pd.DataFrame(res)
        lasso_df.insert(0, "year", int(year))

        est_path = OUT_DIR / f"lasso_estimation_{year}.parquet"
        lasso_df.to_parquet(est_path, index=False)

        mean_nnz_sig = float(lasso_df["nnz_signal"].mean())
        sd_nnz_sig = float(lasso_df["nnz_signal"].std(ddof=0))
        print(f"[{year}] saved estimation: {est_path}")
        print(f"[{year}] nnz_signal mean={mean_nnz_sig:.2f} sd={sd_nnz_sig:.2f} (alpha={alpha_year})")

        class_df = classify_investors_positive_only(
            lasso_df=lasso_df,
            feature_cols_all=feature_cols_all,
            feature_cols_signal=feature_cols_signal
        )
        class_path = OUT_DIR / f"investor_classification_{year}.parquet"
        class_df.to_parquet(class_path, index=False)

        n_class = int(class_df["is_classified"].sum())
        print(f"[{year}] saved classification: {class_path} (classified={n_class:,} / {class_df.shape[0]:,})")

    print(f"\nALL DONE. Total minutes: {(time.time() - t0)/60:.2f}")

if __name__ == "__main__":
    main()