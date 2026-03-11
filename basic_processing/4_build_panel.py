import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# --- 1. Setup & Paths ---
load_dotenv("../.env")
DATA_DIR = Path(os.getenv("DATA_DIR", "/scratch/general/vast/u6025801"))

# Input paths
COMPUSTAT_FILE = DATA_DIR / "data/prices/raw/compustat_whole.parquet"
FACTORS_FILE = DATA_DIR / "data/jkp/jkp_factors_theme.csv"
TRADES_DIR = DATA_DIR / "data/investors/inst_monthly"
LASSO_DIR = DATA_DIR / "data/investors/lasso_out_monthly"
THEME_FILE = LASSO_DIR / "jkp_full_theme.parquet"

# Output path
OUT_FILE = LASSO_DIR / "data_to_j2.csv"

# Time config
START_YEAR = 2010
END_YEAR = 2023

# --- 2. Process Factor Returns ---
print("Processing Factor Returns...")
factors = pd.read_csv(FACTORS_FILE)
factors['date0'] = pd.to_datetime(factors['date'])
factors['year'] = factors['date0'].dt.year
factors['month'] = factors['date0'].dt.month
factors['mdate'] = factors['year'] * 100 + factors['month']

# Sort to prepare for lags/leads later
factors = factors.sort_values(['name', 'mdate']).reset_index(drop=True)
factors = factors[['name', 'year', 'month', 'ret', 'mdate']]

# --- 3. Process Compustat ---
print("Processing Compustat...")
comp = pd.read_parquet(COMPUSTAT_FILE)
comp['year'] = comp['datadate'].dt.year
comp['month'] = comp['datadate'].dt.month
comp['yearmo'] = comp['year'] * 100 + comp['month']
comp = comp.dropna(subset=['isin'])

# Groupby aggregations
comp['totvol'] = comp.groupby(['isin', 'yearmo'])['cshtrd'].transform('sum')
comp['shrout'] = comp.groupby(['isin', 'yearmo'])['cshoc'].transform('median')

# Keep last observation per isin-yearmo
comp = comp.sort_values(['isin', 'yearmo', 'datadate'])
comp = comp.drop_duplicates(subset=['isin', 'yearmo'], keep='last')

# Calculate market cap and lags
comp['mktcap'] = comp['cshoc'] * comp['prccd']
comp['mktcap_l1'] = comp.groupby('isin')['mktcap'].shift(1)

# Calculate cross-sectional weights
comp['sum_mktcap_l1'] = comp.groupby('yearmo')['mktcap_l1'].transform('sum')
comp['w'] = comp['mktcap_l1'] / comp['sum_mktcap_l1']
comp['sqrt_w'] = np.sqrt(comp['w'])

# Filter years and keep columns
comp = comp[(comp['year'] >= START_YEAR) & (comp['year'] <= END_YEAR)]
comp = comp[['isin', 'year', 'month', 'shrout', 'totvol', 'mktcap_l1', 'w', 'sqrt_w']]

# --- 4. Process Themes ---
print("Loading Themes...")
themes = pd.read_parquet(THEME_FILE)

# Mapping dictionary for the signals exactly as defined in Stata
signal_map = {
    "accruals": "rank_Accruals_l1",
    "debt_issuance": "rank_Debt_Issuance_l1",
    "investment": "rank_Investment_l1",
    "low_leverage": "rank_Low_Leverage_l1",
    "low_risk": "rank_Low_Risk_l1",
    "momentum": "rank_Momentum_l1",
    "profit_growth": "rank_Profit_Growth_l1",
    "profitability": "rank_Profitability_l1",
    "quality": "rank_Quality_l1",
    "seasonality": "rank_Seasonality_l1",
    "short_term_reversal": "rank_Short_Term_Reversal_l1",
    "size": "rank_Size_l1",
    "value": "rank_Value_l1"
}

# --- 5. The Big Loop (Yearly Processing) ---
all_years_data = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"[{year}] Processing...")
    
    # Load classifications
    class_file = LASSO_DIR / f"investor_classification_{year}.parquet"
    if not class_file.exists():
        continue
        
    inv_class = pd.read_parquet(class_file)
    inv_class = inv_class[inv_class['is_classified'] == 1].copy()
    
    # Regex replacement for 'name'
    inv_class['name'] = inv_class['top_feature'].str.replace(r"^b_rank_", "", regex=True).str.replace(r"_l1$", "", regex=True)
    inv_class['name'] = inv_class['name'].str.strip().str.lower()
    inv_class = inv_class[['encryp_acnt_no', 'year', 'name']]
    
    # Load and append trades for the year
    trade_files = glob.glob(str(TRADES_DIR / f"inst_trades_monthly_{year}*.parquet"))
    if not trade_files:
        continue
        
    trades = pd.concat([pd.read_parquet(f) for f in trade_files], ignore_index=True)
    
    # Merge classifications
    merged = trades.merge(inv_class, on=['encryp_acnt_no', 'year'], how='inner')
    
    # Collapse (sum) trdvol by name, isin, year, month
    collapsed = merged.groupby(['name', 'isin', 'year', 'month'], as_index=False)['trdvol'].sum()
    
    # Merge themes
    collapsed = collapsed.merge(themes, on=['isin', 'year', 'month'], how='inner')
    
    # Dynamically extract the correct signal based on 'name'
    def get_signal(row):
        col = signal_map.get(row['name'])
        return row[col] if col and col in row else 0

    collapsed['signal'] = collapsed.apply(get_signal, axis=1)
    
    # Merge compustat
    collapsed = collapsed.merge(comp, on=['isin', 'year', 'month'], how='inner')
    
    # Calculate trdvol_signal
    collapsed['trdvol_signal'] = (collapsed['trdvol'] / collapsed['shrout']) * (collapsed['signal'] * 2)
    
    # Weighted mean [aw = w]
    def wavg(group, avg_name, weight_name):
        d = group[avg_name]
        w = group[weight_name]
        try:
            return (d * w).sum() / w.sum()
        except ZeroDivisionError:
            return np.nan

    # Apply weighted mean for trd_intensity
    final_year = collapsed.groupby(['name', 'year', 'month']).apply(wavg, 'trdvol_signal', 'w').reset_index(name='trd_intensity')
    
    all_years_data.append(final_year)

# --- 6. Post-Processing & Cumulatives ---
print("Combining years and calculating cumulative metrics...")
panel = pd.concat(all_years_data, ignore_index=True)

# Merge with factors
panel = panel.merge(factors, on=['name', 'year', 'month'], how='inner')
panel = panel.sort_values(['name', 'mdate']).reset_index(drop=True)

# Helper function to calculate compounded returns
def compound_ret(series, window):
    return (1 + series).rolling(window).apply(np.prod, raw=True) - 1

# Calculate cumulative lags and leads per factor (name)
def calc_cums(g):
    # Returns (Compounded)
    g['ret_l1_l12'] = compound_ret(g['ret'], 12).shift(1)
    g['ret_l1_l24'] = compound_ret(g['ret'], 24).shift(1)
    g['ret_l1_l36'] = compound_ret(g['ret'], 36).shift(1)
    g['ret_l13_l24'] = compound_ret(g['ret'], 12).shift(13)
    g['ret_l25_l36'] = compound_ret(g['ret'], 12).shift(25)
    
    g['ret_f1_f12'] = compound_ret(g['ret'], 12).shift(-12)
    g['ret_f1_f24'] = compound_ret(g['ret'], 24).shift(-24)
    g['ret_f1_f36'] = compound_ret(g['ret'], 36).shift(-36)
    g['ret_f13_f24'] = compound_ret(g['ret'], 12).shift(-24)
    g['ret_f25_f36'] = compound_ret(g['ret'], 12).shift(-36)
    
    # Trading Intensities (Sums)
    g['trd_intensity_l1_l12'] = g['trd_intensity'].rolling(12).sum().shift(1)
    g['trd_intensity_l1_l24'] = g['trd_intensity'].rolling(24).sum().shift(1)
    g['trd_intensity_l1_l36'] = g['trd_intensity'].rolling(36).sum().shift(1)
    g['trd_intensity_l13_l24'] = g['trd_intensity'].rolling(12).sum().shift(13)
    g['trd_intensity_l25_l36'] = g['trd_intensity'].rolling(12).sum().shift(25)
    
    g['trd_intensity_f1_f12'] = g['trd_intensity'].rolling(12).sum().shift(-12)
    g['trd_intensity_f1_f24'] = g['trd_intensity'].rolling(24).sum().shift(-24)
    g['trd_intensity_f1_f36'] = g['trd_intensity'].rolling(36).sum().shift(-36)
    g['trd_intensity_f13_f24'] = g['trd_intensity'].rolling(12).sum().shift(-24)
    g['trd_intensity_f25_f36'] = g['trd_intensity'].rolling(12).sum().shift(-36)
    
    return g

print("Applying lag/lead aggregations...")
panel = panel.groupby('name', group_keys=False).apply(calc_cums)

# --- 7. Save Final Output ---
print(f"Saving final dataset to {OUT_FILE}...")
panel.to_csv(OUT_FILE, index=False)
print("Complete!")