# ---  DOWNLOAD data from wrds and do basic processing

import pandas as pd
import wrds
import datetime
from datetime import date
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import numpy as np
import time
from pathlib import Path
import os
from joblib import Parallel, delayed
import multiprocessing as mp
import polars as pl
from dotenv import load_dotenv

# remove multiprocessing.resource_tracker warnings
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
)

# load directories
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR"))
CODE_DIR = Path(os.getenv("CODE_DIR"))

# %% connect to WRDS. https://wrds-www.wharton.upenn.edu/documents/1443/wrds_connection.html

# conn.list_libraries()
# cc = conn.list_tables(library='comp_global_daily')
# [x for x in cc if 'g_secd' in x]


# %%
# ===================================================================
#                    daily stock data and save in monthly partitions
# ===================================================================

conn = wrds.Connection()

# get all korean stocks in a year
tt = time.time()
this_year = 2024
query = f"""SELECT datadate, 
                    gvkey, 
                    iid, 
                    isin, 
                    conm, 
                    curcdd, 
                    ajexdi, 
                    cshoc, 
                    cshtrd, 
                    prcld, 
                    prchd, 
                    prccd, 
                    divd, 
                    gind 
                FROM comp_global_daily.g_secd 
                WHERE fic = 'KOR' 
                AND datadate BETWEEN '{this_year}-01-01' and '{this_year}-12-31'"""
data = conn.raw_sql(query)
data = pl.from_pandas(data)
tt = time.time() - tt
print(f"Time to get all korean stocks: {tt:.2f} seconds")

# some basic formatting
data = data.with_columns(
    (pl.col("datadate").str.to_date("%Y-%m-%d").alias("datadate"))
)

# yyyymm
data = data.with_columns(
    (pl.col("datadate").dt.strftime("%Y%m").cast(pl.Int32).alias("yyyymm"))
)

# save by month
yms = sorted(data.select('yyyymm').unique().to_series().to_list())

to_dir = DATA_DIR / 'data/prices/daily_compustat/'
to_dir.mkdir(parents=True, exist_ok=True)
for this_ym in yms:
    out_dir = f"{to_dir}/yyyymm={this_ym}"
    os.makedirs(out_dir, exist_ok=True)
    data.filter(pl.col("yyyymm") == this_ym).drop('yyyymm').write_parquet(f"{out_dir}/part.parquet", compression="snappy")


# # ---- example: reading data

from_dir = DATA_DIR / 'data/prices/daily_compustat/'

data = pl.scan_parquet(from_dir).filter(pl.col("yyyymm").is_in([202401, 202402]))
data = data.collect()


