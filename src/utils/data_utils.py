"""
Data utilities for VAHAN web scraper.
Provides data validation, sample data generation, and helper functions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

from ..core.config import Config
from ..core.exceptions import ValidationError
from ..utils.logging_utils import get_logger

def create_sample_data() -> pd.DataFrame:
    """
    MAXED 2025: Create rich, realistic synthetic VAHAN-style registration data
    for testing ETL, dashboards, ML pipelines, or scraper validation.

    Returns:
        pd.DataFrame: Fully structured synthetic VAHAN dataset
    """
    # --- Seeding for stable reproducibility ---
    np.random.seed(42)
    random.seed(42)

    # --- Master domain values ---
    states = [
        'Karnataka', 'Maharashtra', 'Delhi', 'Tamil Nadu',
        'Gujarat', 'Uttar Pradesh', 'West Bengal'
    ]
    years = [2021, 2022, 2023, 2024]

    vehicle_classes = [
        'MOTOR CYCLE', 'SCOOTER', 'CAR', 'AUTO RICKSHAW',
        'TRUCK', 'BUS', 'TEMPO', 'TRACTOR'
    ]

    # --- Output container ---
    sample_data = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for state in states:
        state_code_num = np.random.randint(10, 99)

        for year in years:
            # Realistic YoY multipliers
            year_growth = {
                2021: 1.00,
                2022: np.random.uniform(1.05, 1.15),
                2023: np.random.uniform(1.08, 1.20),
                2024: np.random.uniform(1.10, 1.25),
            }[year]

            for vehicle_class in vehicle_classes:

                # --- Base registrations with variance ---
                base_reg = np.random.randint(1200, 60000)
                registrations = int(base_reg * year_growth)

                # --- Category numbers by vehicle type ---
                is_2w = vehicle_class in ["MOTOR CYCLE", "SCOOTER"]
                is_3w = vehicle_class in ["AUTO RICKSHAW", "TEMPO"]
                is_lmv = vehicle_class in ["CAR", "TEMPO"]
                is_mmv = vehicle_class in ["TRUCK", "BUS"]
                is_hmv = vehicle_class in ["TRUCK", "BUS", "TRACTOR"]

                row = {
                    "S No": len(sample_data) + 1,
                    "Vehicle Class": vehicle_class,

                    # 2W categories
                    "2WIC": np.random.randint(100, 1200) if is_2w else 0,
                    "2WN" : np.random.randint(800, 6000)  if is_2w else 0,
                    "2WT" : np.random.randint(200, 2200)  if is_2w else 0,

                    # 3W categories
                    "3WN" : np.random.randint(150, 1700)  if is_3w else 0,
                    "3WT" : np.random.randint(80, 950)    if is_3w else 0,

                    # LMV/MMV/HMV
                    "LMV": np.random.randint(300, 4000)   if is_lmv else 0,
                    "MMV": np.random.randint(80, 1200)    if is_mmv else 0,
                    "HMV": np.random.randint(40, 800)     if is_hmv else 0,

                    # Total registration number
                    "TOTAL": registrations,

                    # Filter metadata
                    "Filter_State": f"{state}({state_code_num})",
                    "Filter_Year": str(year),
                    "Filter_Vehicle_Type": "ALL",

                    # Timestamp
                    "Scraped_Date": now_str,
                }

                sample_data.append(row)

    df = pd.DataFrame(sample_data)

    # --- Shuffle rows to mimic real scraped randomness ---
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

def validate_data_format(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the structure, completeness, and quality of VAHAN dataset.
    
    Args:
        data (pd.DataFrame): DataFrame to validate.

    Returns:
        Tuple[bool, List[str]]: (is_valid, issues_list)
    """
    logger = get_logger(__name__)
    issues: List[str] = []

    # --- Basic checks --------------------------------------------------------
    if data.empty:
        issues.append("DataFrame is empty")
        return False, issues

    # --- Required columns check ---------------------------------------------
    required_columns = ['TOTAL']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")

    # --- Numeric column validation ------------------------------------------
    numeric_cols = [col for col in Config.NUMERIC_COLUMNS if col in data.columns]
    if not numeric_cols:
        issues.append("No expected numeric columns present")
    else:
        for col in numeric_cols:
            try:
                pd.to_numeric(data[col], errors="coerce")
            except Exception as exc:
                issues.append(f"Column `{col}` cannot be interpreted as numeric: {exc}")

    # --- Temporal columns ----------------------------------------------------
    temporal_cols = [c for c in data.columns 
                     if any(key in c.lower() for key in ("year", "date", "time"))]
    if not temporal_cols:
        issues.append("No temporal columns detected (year, date, time)")

    # --- Location/state columns ---------------------------------------------
    location_cols = [c for c in data.columns
                     if any(key in c.lower() for key in ("state", "location", "region"))]
    if not location_cols:
        issues.append("No location columns found (state, region)")

    # --- Missing data check --------------------------------------------------
    total_rows = len(data)
    for col in data.columns:
        missing_ratio = data[col].isna().mean() * 100
        if missing_ratio > 50:
            issues.append(f"Column `{col}` has {missing_ratio:.1f}% missing values")

    # --- Duplicate rows ------------------------------------------------------
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")

    # --- Log results ---------------------------------------------------------
    if issues:
        logger.warning(f"⚠️ Validation finished with {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues

    logger.info("✅ Data validation passed — no issues found")
    return True, []

def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names using intelligent matching.
    
    Handles:
    - Case-insensitive matching
    - Underscore / space / hyphen normalization
    - Common alias mapping
    - Consistent final naming (PascalCase or defined mappings)
    """
    df = data.copy()

    # --- Canonical target column names -----------------------------------------
    canonical_map = {
        "total": "TOTAL",
        "grandtotal": "TOTAL",
        "grand_total": "TOTAL",

        "state": "State",
        "statename": "State",
        "state_name": "State",
        "filterstate": "Filter_State",

        "year": "Year",
        "filteryear": "Filter_Year",

        "vehicleclass": "Vehicle Class",
        "vehicle_class": "Vehicle Class",
        "vehicletype": "Vehicle Class",
        "vehicle type": "Vehicle Class",
    }

    def normalize_key(col: str) -> str:
        """Normalize a column name into a comparable token."""
        return (
            col.replace(" ", "")
               .replace("_", "")
               .replace("-", "")
               .strip()
               .lower()
        )

    # --- Apply intelligent normalization ---------------------------------------
    new_names = {}

    for col in df.columns:
        key = normalize_key(col)

        # If a canonical mapping exists — use it
        if key in canonical_map:
            new_names[col] = canonical_map[key]
        else:
            # Otherwise standardize format: PascalCase (or keep original)
            parts = re.split(r"[_\-\s]+", col.strip())
            new_names[col] = " ".join(p.capitalize() for p in parts if p)

    return df.rename(columns=new_names)

def detect_data_source(data: pd.DataFrame) -> str:
    """Intelligently detect the source/type of VAHAN data using multi-signal rules.
    
    Detection hierarchy:
    - Strong signals (Scraped_Date, system columns)
    - Structural patterns (Filter_*, category columns)
    - Weak signals (column similarity, numeric profiles)
    - Fallback = Unknown
    """

    df = data.copy()
    cols = set(df.columns)
    lower_cols = {c.lower(): c for c in cols}  # lowercase mapped to real

    # --- Strong Signals -------------------------------------------------------
    if "Scraped_Date" in cols:
        return "Live Scraped Data"

    if any(c.startswith("Filter_") for c in cols):
        return "Dashboard Export"

    if "Vehicle_Category" in cols or "vehicle_category" in lower_cols:
        return "Processed Data"

    # --- Structural Signals ---------------------------------------------------
    raw_required = {"vehicle class", "total"}
    normalized = {c.lower().replace("_", " ").strip() for c in cols}

    if raw_required.issubset(normalized):
        # Check if it's "pure" raw data (no filters, no processed columns)
        processed_like = any(
            kw in normalized
            for kw in ["category", "segment", "cleaned", "normalized"]
        )
        if not processed_like:
            return "Raw VAHAN Data"

    # --- Pattern-Based Heuristics --------------------------------------------
    # Detect Excel/CSV exports (common prefixes/sheet-like names)
    exporting_patterns = ["sheet", "export", "downloaded", "x_", "index"]
    if any(any(p in c.lower() for p in exporting_patterns) for c in cols):
        return "CSV/Excel Export"

    # Detect user-enriched datasets
    user_added_signals = ["source", "created_by", "meta", "scraper_version"]
    if any(s in normalized for s in user_added_signals):
        return "User-Enhanced Data"

    # If lots of numeric operational columns → indicates raw dashboard
    numeric_like = [
        c for c in cols
        if c.lower() not in ["vehicle class", "state", "year"]
        and df[c].dtype.kind in "if"
    ]
    if len(numeric_like) >= 5:
        return "Likely Raw VAHAN Data"

    # --- Weak Heuristic Matching ----------------------------------------------
    # If columns relate to trend analysis, pivot, aggregated views
    agg_keywords = ["growth", "trend", "pivot", "rolling", "avg", "rate"]
    if any(any(k in c.lower() for k in agg_keywords) for c in cols):
        return "Aggregated / Analytics Data"

    # --------------------------------------------------------------------------
    return "Unknown Source"

def calculate_data_quality_score(data: pd.DataFrame) -> Dict:
    """
    Advanced data quality scoring with multi-dimensional metrics:
    - Completeness
    - Consistency
    - Validity
    - Uniqueness
    - Outlier detection
    - Schema stability

    Returns detailed scoring + issues.
    """
    metrics = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "completeness_score": 0,
        "consistency_score": 0,
        "validity_score": 0,
        "uniqueness_score": 0,
        "schema_score": 0,
        "outlier_score": 0,
        "overall_score": 0,
        "issues": [],
    }

    # ----------------------------------------------------------------------
    # 🔴 Empty dataset
    # ----------------------------------------------------------------------
    if data.empty:
        metrics["issues"].append("Dataset is empty – cannot evaluate quality")
        metrics["overall_score"] = 0
        return metrics

    # ======================================================================
    # 1️⃣ COMPLETENESS (Missing values)
    # ======================================================================
    total_cells = data.size
    non_null_cells = data.count().sum()
    completeness = (non_null_cells / total_cells) * 100
    metrics["completeness_score"] = round(completeness, 2)

    missing_per_column = data.isnull().mean() * 100
    for col, pct in missing_per_column.items():
        if pct > 30:
            metrics["issues"].append(f"Column '{col}' has high missing values ({pct:.1f}%)")

    # ======================================================================
    # 2️⃣ CONSISTENCY (Data type correctness)
    # ======================================================================
    consistency_penalty = 0

    for col in data.columns:
        col_data = data[col]

        # Detect numeric columns automatically (not only Config.NUMERIC_COLUMNS)
        if col_data.dtype.kind in "iuf" or col.lower() in {c.lower() for c in Config.NUMERIC_COLUMNS}:
            try:
                pd.to_numeric(col_data, errors="raise")
            except:
                consistency_penalty += 1
                metrics["issues"].append(f"Inconsistent numeric format in '{col}'")

        # Detect datetime columns
        if any(key in col.lower() for key in ["date", "time", "year"]):
            try:
                pd.to_datetime(col_data, errors="raise")
            except:
                metrics["issues"].append(f"Inconsistent datetime format in '{col}'")
                consistency_penalty += 1

    metrics["consistency_score"] = max(0, 100 - consistency_penalty * 8)

    # ======================================================================
    # 3️⃣ VALIDITY (Range checks + logical correctness)
    # ======================================================================
    validity_penalty = 0

    if "TOTAL" in data.columns:
        # Invalid negatives
        negatives = (data["TOTAL"] < 0).sum()
        if negatives > 0:
            validity_penalty += 1
            metrics["issues"].append(f"{negatives} negative TOTAL values found")

        # Extreme values (higher threshold auto-calculated via IQR)
        q1, q3 = data["TOTAL"].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_threshold = q3 + 3 * iqr

        extreme = (data["TOTAL"] > upper_threshold).sum()
        if extreme > 0:
            validity_penalty += 1
            metrics["issues"].append(
                f"{extreme} extreme TOTAL values detected (>{int(upper_threshold)})"
            )

    metrics["validity_score"] = max(0, 100 - validity_penalty * 12)

    # ======================================================================
    # 4️⃣ UNIQUENESS (Duplicate rows)
    # ======================================================================
    duplicate_count = data.duplicated().sum()
    duplicate_pct = (duplicate_count / len(data)) * 100
    metrics["uniqueness_score"] = max(0, 100 - duplicate_pct)

    if duplicate_count > 0:
        metrics["issues"].append(f"{duplicate_count} duplicate rows detected")

    # ======================================================================
    # 5️⃣ SCHEMA STABILITY (Column naming quality)
    # ======================================================================
    schema_penalty = 0

    for col in data.columns:
        if not col.strip():
            schema_penalty += 1
            metrics["issues"].append("Unnamed/empty column detected")
            continue

        # Too many special characters → low schema quality
        if any(ch in col for ch in ["$", "%", "@", "/", "\\"]):
            schema_penalty += 1
            metrics["issues"].append(f"Suspicious characters in column '{col}'")

    metrics["schema_score"] = max(0, 100 - schema_penalty * 10)

    # ======================================================================
    # 6️⃣ OUTLIER SCORE (Across all numeric columns)
    # ======================================================================
    numeric_cols = [c for c in data.columns if data[c].dtype.kind in "if"]
    outlier_penalty = 0

    for col in numeric_cols:
        col_series = data[col].dropna()
        if col_series.empty:
            continue

        q1, q3 = col_series.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        lower = q1 - 3 * iqr

        outliers = ((col_series > upper) | (col_series < lower)).sum()

        if outliers > 0:
            outlier_penalty += 1
            metrics["issues"].append(f"Column '{col}' contains {outliers} outliers")

    metrics["outlier_score"] = max(0, 100 - outlier_penalty * 5)

    # ======================================================================
    # 7️⃣ OVERALL SCORE (Weighted)
    # ======================================================================
    overall = (
        metrics["completeness_score"] * 0.30
        + metrics["consistency_score"] * 0.20
        + metrics["validity_score"] * 0.20
        + metrics["uniqueness_score"] * 0.10
        + metrics["schema_score"] * 0.10
        + metrics["outlier_score"] * 0.10
    )
    metrics["overall_score"] = round(overall, 2)

    return metrics

def generate_data_summary(data: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive, production-grade summary of the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        Dict: Complete dataset summary with structure, quality, content & metadata
    """
    if data is None or not isinstance(data, pd.DataFrame):
        return {
            "error": "Invalid input — expected a pandas DataFrame",
            "basic_info": {},
            "columns": {},
            "data_quality": {},
            "content_summary": {}
        }

    # ----------------------------------------------------------------------
    # SAFE COPY
    # ----------------------------------------------------------------------
    data = data.copy()

    # Handle datetime conversion safely
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col], errors='ignore')
            except:
                pass

    summary = {
        "basic_info": {},
        "columns": {},
        "data_quality": {},
        "content_summary": {},
        "advanced_stats": {}
    }

    # ----------------------------------------------------------------------
    # BASIC INFO
    # ----------------------------------------------------------------------
    try:
        memory_mb = round(data.memory_usage(deep=True).sum() / (1024**2), 2)
    except:
        memory_mb = "N/A"

    summary["basic_info"] = {
        "rows": len(data),
        "columns": len(data.columns),
        "memory_usage_mb": memory_mb,
        "data_source": detect_data_source(data)
    }

    # ----------------------------------------------------------------------
    # COLUMN TYPES
    # ----------------------------------------------------------------------
    summary["columns"] = {
        "all_columns": data.columns.tolist(),
        "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
        "text_columns": data.select_dtypes(include=["object"]).columns.tolist(),
        "datetime_columns": data.select_dtypes(include=["datetime64", "datetime"]).columns.tolist(),
        "missing_values": data.isna().sum().to_dict(),
        "column_uniques": {col: data[col].nunique() for col in data.columns}
    }

    # ----------------------------------------------------------------------
    # DATA QUALITY
    # ----------------------------------------------------------------------
    try:
        summary["data_quality"] = calculate_data_quality_score(data)
    except Exception as e:
        summary["data_quality"] = {"error": f"Quality score failed: {str(e)}"}

    # ----------------------------------------------------------------------
    # CONTENT SUMMARY
    # ----------------------------------------------------------------------
    content = {}

    # TOTAL column summary
    if "TOTAL" in data.columns and pd.api.types.is_numeric_dtype(data["TOTAL"]):
        content["total_registrations"] = int(data["TOTAL"].sum())
        content["avg_registrations"] = round(data["TOTAL"].mean(), 2)
        content["min_total"] = int(data["TOTAL"].min())
        content["max_total"] = int(data["TOTAL"].max())

    # State summary
    if "State" in data.columns:
        content["unique_states"] = data["State"].nunique()
        content["top_states"] = data["State"].value_counts().head(10).to_dict()

    # Year summary
    if "Year" in data.columns:
        try:
            content["year_range"] = f"{int(data['Year'].min())}-{int(data['Year'].max())}"
        except:
            content["year_range"] = "Invalid year values"

    # Vehicle Class summary
    if "Vehicle Class" in data.columns:
        content["unique_vehicle_classes"] = data["Vehicle Class"].nunique()
        content["top_vehicle_classes"] = data["Vehicle Class"].value_counts().head(10).to_dict()

    summary["content_summary"] = content

    # ----------------------------------------------------------------------
    # ADVANCED STATISTICS (MAXED)
    # ----------------------------------------------------------------------
    advanced = {}

    # Outlier detection for numeric columns
    numeric_cols = summary["columns"]["numeric_columns"]
    outlier_info = {}

    for col in numeric_cols:
        try:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = data[(data[col] < lower) | (data[col] > upper)].shape[0]
            outlier_info[col] = outliers
        except:
            outlier_info[col] = "N/A"

    advanced["numeric_outliers"] = outlier_info

    # Correlation matrix (safe)
    try:
        corr = data[numeric_cols].corr().round(3).to_dict()
    except:
        corr = {}

    advanced["correlation_matrix"] = corr

    # Missing data percentage
    advanced["missing_percentage"] = (
        (data.isna().sum() / len(data) * 100).round(2).to_dict()
        if len(data) > 0 else {}
    )

    summary["advanced_stats"] = advanced

    return summary
