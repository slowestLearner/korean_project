import polars as pl
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv("../.env")
DATA_DIR = Path(os.getenv("DATA_DIR"))
CODE_DIR = Path(os.getenv("CODE_DIR"))

# --- Configuration & Paths ---
# Data is read from and written to the scratch drive (DATA_DIR)
base_dir = DATA_DIR / "data/raw/parsed_trades" 
ksq_dir = base_dir / "ksq"
stk_dir = base_dir / "stk"

out_dir = DATA_DIR / "data/investors/inst_monthly"
out_dir.mkdir(parents=True, exist_ok=True)

# Find all monthly partition folders (e.g., "yyyymm=201001")
partition_folders = sorted([f for f in os.listdir(ksq_dir) if f.startswith("yyyymm=")])

tt_total = time.time()

for folder in partition_folders:
    yyyymm = folder.split("=")[1]
    print(f"Processing month: {yyyymm}...")
    tt = time.time()
    
    # Paths for this specific month's part.parquet
    ksq_file = ksq_dir / folder / "part.parquet"
    stk_file = stk_dir / folder / "part.parquet"
    
    # Load and concatenate KSQ and STK data for the month
    dfs = []
    if ksq_file.exists():
        dfs.append(pl.read_parquet(ksq_file))
    if stk_file.exists():
        dfs.append(pl.read_parquet(stk_file))
        
    if not dfs:
        print(f"  No data found for {yyyymm}, skipping.")
        continue
        
    df = pl.concat(dfs, how="vertical")
    
    # --- 1. Filter by Investor Type ---
    df = df.filter(pl.col("invst_tp_cd") < 8000)
    
    # --- 2. Calculate Net and All Trades ---
    df = df.with_columns([
        (pl.col("trdvol") * pl.col("trd_prc")).alias("trd_amt"),
        (pl.col("trdvol").abs() * pl.col("trd_prc")).alias("abs_trd_amt")
    ])
    
    # Drop if trd_amt == 0 or missing
    df = df.filter((pl.col("trd_amt") != 0) & pl.col("trd_amt").is_not_null())
    
    # --- 3. Extract Dates ---
    df = df.with_columns([
        (pl.col("trd_dd") // 10000).cast(pl.Int32).alias("year"),
        ((pl.col("trd_dd") // 100) % 100).cast(pl.Int32).alias("month")
    ])
    
    # --- 4. Collapse (sum) ---
    df_collapsed = df.group_by(["encryp_acnt_no", "isu_cd", "year", "month"]).agg([
        pl.col("trdvol").sum(),
        pl.col("trd_amt").sum(),
        pl.col("abs_trd_amt").sum()
    ])
    
    # --- 5. Trading Intensity & Standardization ---
    group_cols = ["year", "month", "encryp_acnt_no"]
    
    df_final = df_collapsed.with_columns([
        pl.col("abs_trd_amt").sum().over(group_cols).alias("sum_abs_trd_amt"),
        pl.len().over(group_cols).alias("num") 
    ]).with_columns([
        (pl.col("trd_amt") / pl.col("sum_abs_trd_amt")).alias("trade")
    ]).with_columns([
        ((pl.col("trade") - pl.col("trade").mean().over(group_cols)) / 
          pl.col("trade").std(ddof=1).over(group_cols)).alias("trade_std")
    ])
    
    # --- 6. Formatting and Saving ---
    df_final = df_final.rename({"isu_cd": "isin"}).select([
        "year", "month", "isin", "encryp_acnt_no", "trade", "trade_std", "trdvol", "num"
    ]).sort(["year", "month", "isin", "encryp_acnt_no"])
    
    # Save the file
    out_file = out_dir / f"inst_trades_monthly_{yyyymm}.parquet"
    df_final.write_parquet(out_file)
    
    print(f"  -> Saved {out_file.name}. Time taken: {((time.time() - tt) / 60):.2f} mins")

print(f"\nAll done! Total time taken: {((time.time() - tt_total) / 60):.2f} mins")

