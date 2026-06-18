"""
Real Estate Price Predictor — Data Preparation Pipeline
FutureAI Analytics | Housing Transactions Dataset
=======================================================
Steps:
  1. Chunked data loading (OOM-safe)
  2. Exploratory Data Analysis (EDA) + Seaborn plots
  3. Outlier removal (±3 std deviations)
  4. Feature engineering (3 new features)
  5. Save to optimized .parquet

Run:
    pip install pandas numpy seaborn matplotlib pyarrow faker
    python real_estate_data_prep.py
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RAW_CSV      = "housing_transactions.csv"   # swap with your real 50 GB file
PARQUET_OUT  = "housing_cleaned.parquet"
CHUNK_SIZE   = 100_000                      # rows per chunk (~OOM-safe for 50 GB)
PLOT_DIR     = "eda_plots"
CITY_LAT     = 40.7128                      # e.g. New York City center
CITY_LON     = -74.0060
CURRENT_YEAR = 2026
STD_THRESHOLD = 3
RANDOM_SEED  = 42

os.makedirs(PLOT_DIR, exist_ok=True)


# ─── STEP 0 : Generate a realistic synthetic dataset (demo / CI use) ─────────
def generate_synthetic_csv(path: str, n_rows: int = 500_000) -> None:
    """Creates a messy CSV that mimics a real-world dirty housing dataset."""
    print(f"[DEMO] Generating {n_rows:,} synthetic rows → {path}")
    rng = np.random.default_rng(RANDOM_SEED)

    n = n_rows
    year_built   = rng.integers(1900, 2023, n)
    bedrooms     = rng.integers(1, 7, n).astype(float)
    sqft         = rng.integers(500, 6000, n).astype(float)
    lat          = rng.uniform(40.5, 40.9, n)
    lon          = rng.uniform(-74.3, -73.7, n)

    # Base price with realistic correlations
    price = (
        80_000
        + sqft * 150
        + bedrooms * 12_000
        + (CURRENT_YEAR - year_built) * -300
        + rng.normal(0, 25_000, n)
    ).clip(50_000)

    # ── Inject dirt ──────────────────────────────────────────────────────────
    # 1. Price outliers (0.5 %)
    outlier_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    price[outlier_idx] *= rng.uniform(5, 20, len(outlier_idx))

    # 2. Missing values (~3 %)
    for col_arr in [bedrooms, sqft, lat, lon]:
        miss_idx = rng.choice(n, size=int(n * 0.03), replace=False)
        col_arr[miss_idx] = np.nan

    # 3. Nonsense year_built values
    bad_year = rng.choice(n, size=int(n * 0.01), replace=False)
    year_built[bad_year] = rng.integers(1700, 1850, len(bad_year))

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "price":          price,
        "sqft":           sqft,
        "bedrooms":       bedrooms,
        "year_built":     year_built,
        "latitude":       lat,
        "longitude":      lon,
        "sale_year":      rng.integers(2014, 2024, n),
    })
    df.to_csv(path, index=False)
    print(f"[DEMO] CSV written ({df.shape[0]:,} rows × {df.shape[1]} cols)\n")


# ─── STEP 1 : Chunked loading ─────────────────────────────────────────────────
def load_in_chunks(path: str, chunksize: int) -> pd.DataFrame:
    """
    Reads a (potentially 50 GB) CSV in chunks to avoid OOM errors.
    Each chunk is lightly cleaned before being appended to a list; the
    list is concatenated once at the end (much faster than repeated concat).
    """
    print(f"[1/5] Loading data in chunks of {chunksize:,} rows …")
    chunks = []

    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        # Per-chunk lightweight cleanup
        chunk.drop_duplicates(subset="transaction_id", inplace=True)
        chunk.dropna(subset=["price"], inplace=True)           # price is mandatory
        chunk = chunk[chunk["price"] > 0]
        chunk = chunk[chunk["year_built"].between(1850, CURRENT_YEAR)]
        chunks.append(chunk)

        if (i + 1) % 5 == 0:
            print(f"    … processed {(i+1)*chunksize:,} rows", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n[1/5] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols after chunk cleaning.")
    return df


# ─── STEP 2 : EDA + Seaborn plots ────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    print("\n[2/5] Running Exploratory Data Analysis …")

    # ── Summary stats ────────────────────────────────────────────────────────
    print("\n── Descriptive Statistics ──")
    print(df.describe().to_string())

    print("\n── Missing Values ──")
    miss = df.isnull().sum()
    print(miss[miss > 0].to_string())

    # ── Plot 1 : Raw price distribution ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("House Price Distribution (Raw)", fontsize=14, fontweight="bold")

    sns.histplot(df["price"], bins=80, kde=True, ax=axes[0], color="#2563EB")
    axes[0].set_title("Price — Full Range")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    axes[0].set_xlabel("Sale Price")

    cap = df["price"].quantile(0.99)
    sns.histplot(df.loc[df["price"] <= cap, "price"], bins=80, kde=True,
                 ax=axes[1], color="#7C3AED")
    axes[1].set_title("Price — 99th Percentile Cap")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    axes[1].set_xlabel("Sale Price")

    plt.tight_layout()
    path1 = os.path.join(PLOT_DIR, "01_price_distribution_raw.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"    Saved → {path1}")

    # ── Plot 2 : Correlation heat-map ─────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, "02_correlation_heatmap.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"    Saved → {path2}")

    # ── Plot 3 : Price vs sqft scatter ────────────────────────────────────────
    sample = df.dropna(subset=["sqft"]).sample(min(10_000, len(df)),
                                                random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=sample, x="sqft", y="price",
                    alpha=0.3, s=12, color="#059669", ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1e3:.0f}K"))
    ax.set_title("Price vs. Square Footage (10 K sample)", fontweight="bold")
    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Sale Price")
    plt.tight_layout()
    path3 = os.path.join(PLOT_DIR, "03_price_vs_sqft.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"    Saved → {path3}")


# ─── STEP 3 : Outlier removal (±3 σ on price) ────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/5] Removing mathematical outliers (price beyond ±3 σ) …")
    before = len(df)

    mean_p = df["price"].mean()
    std_p  = df["price"].std()
    lo, hi = mean_p - STD_THRESHOLD * std_p, mean_p + STD_THRESHOLD * std_p

    df = df[(df["price"] >= lo) & (df["price"] <= hi)].copy()
    removed = before - len(df)
    print(f"    Removed {removed:,} outlier rows  "
          f"({removed/before*100:.2f}% of dataset)")
    print(f"    Price range after: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")

    # ── Plot 4 : Post-clean distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["price"], bins=80, kde=True, ax=ax, color="#DC2626")
    ax.set_title("House Price Distribution (After Outlier Removal)", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.set_xlabel("Sale Price")
    plt.tight_layout()
    path4 = os.path.join(PLOT_DIR, "04_price_distribution_clean.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"    Saved → {path4}")

    return df


# ─── STEP 4 : Feature engineering ────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/5] Engineering new features …")

    # Fill remaining NaNs with column medians before computing features
    df["sqft"]      = df["sqft"].fillna(df["sqft"].median())
    df["bedrooms"]  = df["bedrooms"].fillna(df["bedrooms"].median())
    df["latitude"]  = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())

    # ── Feature 1 : Age of Home ───────────────────────────────────────────────
    df["age_of_home"] = CURRENT_YEAR - df["year_built"]
    df["age_of_home"] = df["age_of_home"].clip(lower=0)
    print("    ✓ age_of_home  = current_year − year_built")

    # ── Feature 2 : Distance to City Center (km) ──────────────────────────────
    df["dist_to_city_center_km"] = haversine_km(
        df["latitude"].values,  df["longitude"].values,
        CITY_LAT,               CITY_LON
    )
    print(f"    ✓ dist_to_city_center_km  (city center: {CITY_LAT}, {CITY_LON})")

    # ── Feature 3 : Price per Square Foot ────────────────────────────────────
    df["price_per_sqft"] = (df["price"] / df["sqft"].replace(0, np.nan)).round(2)
    print("    ✓ price_per_sqft  = price / sqft")

    print(f"\n    DataFrame shape after feature engineering: {df.shape}")
    return df


# ─── STEP 5 : Save to Parquet ─────────────────────────────────────────────────
def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    print(f"\n[5/5] Saving cleaned data to {path} …")

    # Cast dtypes to minimise file size
    int_cols   = ["bedrooms", "year_built", "sale_year", "age_of_home"]
    float_cols = ["price", "sqft", "latitude", "longitude",
                  "dist_to_city_center_km", "price_per_sqft"]

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("int16")
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"    ✓ Saved {len(df):,} rows → {path}  ({size_mb:.1f} MB, Snappy compressed)")
    print("\n── Final column overview ──")
    print(df.dtypes.to_string())
    print(f"\n── Sample rows ──")
    print(df.head(3).to_string())


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Real Estate Price Predictor — Data Prep Pipeline")
    print("  FutureAI Analytics | 2026")
    print("=" * 62)

    # 0. Generate synthetic CSV if none exists (remove for production)
    if not os.path.exists(RAW_CSV):
        generate_synthetic_csv(RAW_CSV, n_rows=500_000)

    # 1. Chunk-load
    df = load_in_chunks(RAW_CSV, CHUNK_SIZE)

    # 2. EDA
    run_eda(df)

    # 3. Outlier removal
    df = remove_outliers(df)

    # 4. Feature engineering
    df = engineer_features(df)

    # 5. Save
    save_to_parquet(df, PARQUET_OUT)

    print("\n✅  Pipeline complete.")
    print(f"    Cleaned data  → {PARQUET_OUT}")
    print(f"    EDA plots     → {PLOT_DIR}/")


if __name__ == "__main__":
    main()
