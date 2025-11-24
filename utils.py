"""Utilities for loading and splitting Data.csv."""
import os
import pandas as pd


def load_stock_dataframe(csv_path="Data.csv"):
    """Load Data.csv; parse/sort by Date if present."""
    if not os.path.exists(csv_path):
        alt = "DATA.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Missing {csv_path} or {alt}")
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").set_index("Date")
    return df

# TODO: AUTOMATICALLY ONLY LOAD THINGS AT LEAST A YEAR INTO THE FUTURE
# OR HOWEVER LONG WE WANT TO PREDICT AHEAD, make it a parameter
def get_train_test_val(splits=(0.7, 0.15, 0.15), csv_path="Data.csv"):
    """Return train, val, test DataFrames from Data.csv using counts or percentages."""
    df = load_stock_dataframe(csv_path)
    n = len(df)
    a, b, c = splits
    if all(isinstance(x, int) for x in (a, b, c)):
        if a + b + c != n:
            raise ValueError(f"Split counts must sum to {n}")
        t, v = int(a), int(b)
    else:
        fa, fb, fc = float(a), float(b), float(c)
        if fa + fb + fc > 1 + 1e-6:
            fa, fb, fc = fa / 100.0, fb / 100.0, fc / 100.0
        if abs((fa + fb + fc) - 1) > 1e-6:
            raise ValueError("Percentage splits must sum to 1 or 100")
        t, v = int(round(fa * n)), int(round(fb * n))
    te = n - t - v
    if min(t, v, te) < 0:
        raise ValueError("Invalid splits")
    return df.iloc[:t], df.iloc[t:t + v], df.iloc[t + v:]
