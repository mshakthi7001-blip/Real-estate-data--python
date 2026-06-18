# Real Estate Price Predictor — Data Preparation Pipeline
**FutureAI Analytics | Housing Transactions Dataset**

A production-ready Python pipeline that ingests a large, dirty housing transactions CSV, cleans it, performs EDA, engineers new features, and exports an optimized `.parquet` file ready for ML model training.

---

##  What It Does

| Step | Task |
|------|------|
| 1 | Loads a 50 GB+ CSV **in chunks** to avoid Out-of-Memory errors |
| 2 | Runs **Exploratory Data Analysis** and saves Seaborn plots |
| 3 | Removes **mathematical outliers** (values beyond ±3 standard deviations) |
| 4 | Engineers **3 new features** to improve model accuracy |
| 5 | Saves cleaned data to a compressed **`.parquet`** file (10× faster than CSV) |

---

## Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy seaborn matplotlib pyarrow
```

### 2. Add your data
Place your CSV in the project folder and update line 22 of the script:
```python
RAW_CSV = "your_file_name.csv"  # default: housing_transactions.csv
```
> **No data?** Leave it as-is — the script auto-generates a 500,000-row synthetic dataset.

### 3. Run
```bash
python real_estate_data_prep.py
```

---

##  Output Files

```
project/
├── real_estate_data_prep.py   # Main pipeline script
├── housing_cleaned.parquet    # cleaned, ML-ready dataset
└── eda_plots/
    ├── 01_price_distribution_raw.png
    ├── 02_correlation_heatmap.png
    ├── 03_price_vs_sqft.png
    └── 04_price_distribution_clean.png
```

---

##  Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `age_of_home` | `current_year − year_built` | Captures depreciation and renovation cycles |
| `dist_to_city_center_km` | Haversine distance to city center | Location premium is a top price predictor |
| `price_per_sqft` | `price ÷ sqft` | Normalized value metric used by all appraisers |

---

##  Configuration

All key settings are at the top of the script:

```python
CHUNK_SIZE    = 100_000   # Rows per chunk (increase on high-RAM machines)
STD_THRESHOLD = 3         # Outlier removal threshold (standard deviations)
CITY_LAT      = 40.7128   # City center latitude  (default: New York City)
CITY_LON      = -74.0060  # City center longitude
CURRENT_YEAR  = 2026      # Used to compute age_of_home
```

---

##  Tech Stack

- **pandas** — chunked loading, data cleaning, feature engineering
- **numpy** — vectorized math (haversine distance, outlier thresholds)
- **seaborn / matplotlib** — EDA visualizations
- **pyarrow** — fast Parquet read/write with Snappy compression
