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

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
CODE_DIR = os.getenv("CODE_DIR")

# %% connect to WRDS. https://wrds-www.wharton.upenn.edu/documents/1443/wrds_connection.html
conn = wrds.Connection(wrds_username="jiacui2")

# %%
# ===================================================================
#                          Some basic downloads
# ===================================================================

# ---------------------------------------------------
# Download daily returns from Compustat global
# ---------------------------------------------------


# %%
# ===================================================================
#                    daily stock data and save in monthly partitions
# ===================================================================

# Use monthly data to filter down to common stocks
data_monthly = pl.read_parquet(
    f"{BASE_DIR}/../../input_data/stocks/prices/monthly/returns.parquet"
)
data_monthly = (
    data_monthly.with_columns(pl.col("date").dt.year().alias("yyyy"))
    .select(["yyyy", "permno"])
    .unique()
    .filter(pl.col("yyyy") >= 1980)
)


to_dir = Path(f"{BASE_DIR}/../../input_data/stocks/prices/daily/")
to_dir.mkdir(parents=True, exist_ok=True)

# loop through years
for this_year in range(1980, 2025):
    print(this_year)

    # why is reading so slow?
    tt = time.time()

    query = f"""SELECT date, permno, openprc, prc, bidlo, askhi, vol, shrout, ret, cfacpr, cfacshr
    FROM crsp.dsf WHERE date BETWEEN '{this_year}-01-01' and '{this_year}-12-31'"""

    data = conn.raw_sql(query)
    data = pl.from_pandas(data).lazy()

    # get right formats (sometimes openprc is empty)
    data = data.with_columns(
        [pl.col("openprc").cast(pl.Float64), pl.col("date").cast(pl.Date)]
    )

    # prcs need to be positive
    data = data.with_columns(
        [pl.col("openprc").abs().alias("openprc"), pl.col("prc").abs().alias("prc")]
    ).sort(["date", "permno"])

    # get subset of common stocks
    permnos_this_year = (
        data_monthly.filter(pl.col("yyyy") == this_year).get_column("permno").to_list()
    )
    data = data.filter(pl.col("permno").is_in(permnos_this_year))

    # important: if a stock has any data in this year, make sure it has full data for all dates
    # there is no need to fill any data forward, because if a date has missing ret, we will not use it
    all_permnos = data.select(pl.col("permno").unique())
    all_dates = data.select(pl.col("date").unique().sort())
    grid = all_permnos.join(all_dates, how="cross")
    data = grid.join(data, on=["permno", "date"], how="left")
    del grid
    data = data.sort(["permno", "date"])

    # write to monthly partitions
    data = data.with_columns(
        (pl.col("date").dt.strftime("%Y%m").cast(pl.Int32).alias("yyyymm"))
    )
    data = data.collect()

    for (yyyymm,), group in data.group_by("yyyymm"):
        out_dir = f"{to_dir}/yyyymm={yyyymm}"
        os.makedirs(out_dir, exist_ok=True)
        group.write_parquet(f"{out_dir}/part.parquet", compression="snappy")

    # Calculate the elapsed time
    tt = time.time() - tt

    # Print the time it took for this step
    print(f"Time to process year {this_year}: {tt:.2f} seconds")


