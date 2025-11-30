"""
Data cleaning utilities for VAHAN vehicle registration data.
Handles data validation, cleaning, and standardization.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union
from datetime import datetime

from ..core.config import Config
from ..core.exceptions import DataProcessingError, ValidationError
from ..utils.logging_utils import get_logger

class DataCleaner:
    """Utility class for cleaning and standardizing VAHAN data."""
    
    def __init__(self):
        """Initialize the DataCleaner with robust logging."""
        from logging import getLogger, StreamHandler, Formatter, DEBUG
    
        self.logger = getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():  # Avoid duplicate handlers
            handler = StreamHandler()
            formatter = Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(DEBUG)  # Maxed: debug-level ready
        self.logger.debug("DataCleaner initialized")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input DataFrame structure and content.
    
        Args:
            data (pd.DataFrame): DataFrame to validate
    
        Returns:
            bool: True if data is valid
    
        Raises:
            ValidationError: If data is None, empty, or fails critical checks
        """
        from pandas.api.types import is_numeric_dtype
    
        if data is None:
            self.logger.error("❌ Data is None")
            raise ValidationError("Data cannot be None")
        
        if data.empty:
            self.logger.error("❌ Data is empty")
            raise ValidationError("Data cannot be empty")
        
        # Check required numeric columns
        numeric_cols_present = [col for col in Config.NUMERIC_COLUMNS if col in data.columns and is_numeric_dtype(data[col])]
        
        if not numeric_cols_present:
            self.logger.warning("⚠️ No standard numeric columns found in data")
        else:
            self.logger.debug(f"Numeric columns detected: {numeric_cols_present}")
        
        # Optional: Check for NaNs in numeric columns
        if numeric_cols_present:
            nan_cols = [col for col in numeric_cols_present if data[col].isna().any()]
            if nan_cols:
                self.logger.warning(f"⚠️ NaNs detected in columns: {nan_cols}")
        
        self.logger.info(f"✅ Data validation passed. Shape: {data.shape}")
        return True
    
    def clean_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric columns by removing commas, handling missing values,
        and converting to proper numeric types.
    
        Args:
            data (pd.DataFrame): DataFrame to clean
    
        Returns:
            pd.DataFrame: DataFrame with cleaned numeric columns
        """
        import numpy as np
        data_copy = data.copy()
    
        for col in Config.NUMERIC_COLUMNS:
            if col not in data_copy.columns:
                self.logger.debug(f"Skipping missing column: {col}")
                continue
    
            try:
                # Ensure column is string first, remove commas
                data_copy[col] = data_copy[col].astype(str).str.replace(',', '', regex=False)
    
                # Normalize common missing/invalid values to NaN
                data_copy[col] = data_copy[col].replace(['', 'nan', 'None', '-', 'NA'], np.nan)
    
                # Convert to numeric, coerce errors to NaN
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
    
                # Fill NaNs with 0 (maxed safe default for numeric processing)
                data_copy[col] = data_copy[col].fillna(0)
    
                self.logger.debug(f"✅ Numeric column cleaned: {col}")
    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to clean column '{col}': {e}")
    
        self.logger.info(f"✅ All numeric columns cleaned. Shape: {data_copy.shape}")
        return data_copy
    
    def clean_text_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text columns by standardizing format, handling missing values,
        and normalizing case for specific columns.
    
        Args:
            data (pd.DataFrame): DataFrame to clean
    
        Returns:
            pd.DataFrame: DataFrame with cleaned text columns
        """
        import numpy as np
        data_copy = data.copy()
    
        # Identify object (text) columns, excluding numeric columns
        text_columns = [col for col in data_copy.select_dtypes(include=['object']).columns 
                        if col not in Config.NUMERIC_COLUMNS]
    
        for col in text_columns:
            try:
                # Convert to string and strip whitespace
                data_copy[col] = data_copy[col].astype(str).str.strip()
    
                # Replace common invalid entries with NaN
                data_copy[col] = data_copy[col].replace(['nan', 'None', '', 'NA'], np.nan)
    
                # Standardize case for key descriptive columns
                if any(keyword in col.lower() for keyword in ['state', 'vehicle', 'city', 'district']):
                    data_copy[col] = data_copy[col].str.title()
                else:
                    # lowercase other text columns for consistency
                    data_copy[col] = data_copy[col].str.lower()
    
                self.logger.debug(f"✅ Cleaned text column: {col}")
    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to clean text column '{col}': {e}")
    
        self.logger.info(f"✅ All text columns cleaned. Shape: {data_copy.shape}")
        return data_copy
    
    def extract_temporal_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and clean temporal information such as Year and State.
    
        Args:
            data (pd.DataFrame): DataFrame to process
    
        Returns:
            pd.DataFrame: DataFrame with extracted temporal data
        """
        import pandas as pd
        import numpy as np
        data_copy = data.copy()
    
        # --- Extract Year ---
        year_columns = [col for col in data_copy.columns if 'year' in col.lower()]
        year_extracted = False
        for col in year_columns:
            try:
                data_copy[col] = data_copy[col].astype(str)
                # Extract 4-digit year
                data_copy['Year'] = data_copy[col].str.extract(r'(\d{4})')[0]
                data_copy['Year'] = pd.to_numeric(data_copy['Year'], errors='coerce')
                year_extracted = True
                self.logger.debug(f"✅ Extracted Year from column: {col}")
                break  # Stop after first valid year column
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to extract Year from '{col}': {e}")
        
        if not year_extracted:
            self.logger.warning("⚠️ No valid Year column found. 'Year' column may contain NaN")
    
        # --- Extract State ---
        state_columns = [col for col in data_copy.columns if 'state' in col.lower()]
        state_extracted = False
        for col in state_columns:
            try:
                data_copy['State'] = data_copy[col].astype(str)
                # Remove parentheses, numbers, and extra whitespace
                data_copy['State'] = data_copy['State'].str.replace(r'\([^)]*\)', '', regex=True)
                data_copy['State'] = data_copy['State'].str.replace(r'\d+', '', regex=True)
                data_copy['State'] = data_copy['State'].str.strip()
                # Optional: title case
                data_copy['State'] = data_copy['State'].str.title()
                state_extracted = True
                self.logger.debug(f"✅ Extracted State from column: {col}")
                break
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to extract State from '{col}': {e}")
        
        if not state_extracted:
            self.logger.warning("⚠️ No valid State column found. 'State' column may contain NaN")
    
        self.logger.info(f"✅ Temporal data extraction completed. Shape: {data_copy.shape}")
        return data_copy
    
    def categorize_vehicle(self, row: pd.Series) -> str:
        """
        MAXED EDITION (2025 Hardened Vehicle Categorizer)
        -------------------------------------------------
        Categorize vehicles into 2W, 3W, 4W+ based on class info or numeric columns.
        Fully noise-tolerant, debug-logged, and fallback-aware.
    
        Args:
            row (pd.Series): DataFrame row containing vehicle information
    
        Returns:
            str: Vehicle category ('2W', '3W', '4W+', 'Other', 'Unknown')
        """
        try:
            # ---------- Step 1: Class-based categorization ----------
            vehicle_class_cols = [col for col in row.index if 'vehicle' in col.lower() and 'class' in col.lower()]
            if vehicle_class_cols:
                vehicle_class = str(row[vehicle_class_cols[0]]).upper()
                for category, keywords in Config.VEHICLE_CATEGORIES.items():
                    if any(keyword.upper() in vehicle_class for keyword in keywords):
                        self.logger.debug(f"✅ Categorized by class column '{vehicle_class_cols[0]}': {category}")
                        return category
    
            # ---------- Step 2: Numeric-based categorization ----------
            numeric_values = {}
            for col in Config.NUMERIC_COLUMNS:
                try:
                    numeric_values[col] = float(row[col]) if pd.notna(row[col]) else 0
                except (ValueError, TypeError):
                    numeric_values[col] = 0
    
            if numeric_values:
                # Define numeric groupings
                two_wheeler_cols = ['2WIC', '2WN', '2WT']
                three_wheeler_cols = ['3WN', '3WT']
                four_wheeler_cols = ['LMV', 'MMV', 'HMV']
    
                totals = {
                    '2W': sum(numeric_values.get(col, 0) for col in two_wheeler_cols),
                    '3W': sum(numeric_values.get(col, 0) for col in three_wheeler_cols),
                    '4W+': sum(numeric_values.get(col, 0) for col in four_wheeler_cols)
                }
    
                max_category = max(totals.items(), key=lambda x: x[1])
                if max_category[1] > 0:
                    self.logger.debug(f"✅ Categorized by numeric totals: {max_category[0]} (value: {max_category[1]})")
                    return max_category[0]
    
            # ---------- Step 3: Fallback for ambiguous data ----------
            # Check if any numeric column has non-zero but ungrouped value
            if any(v > 0 for v in numeric_values.values()):
                self.logger.debug("⚠️ Numeric values present but no group matched, categorizing as 'Other'")
                return 'Other'
    
            # ---------- Step 4: Default unknown ----------
            self.logger.debug("⚠️ Unable to categorize vehicle, defaulting to 'Unknown'")
            return 'Unknown'
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error categorizing vehicle: {e}")
            return 'Unknown'
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MAXED EDITION (2025 Hardened Deduplicator)
        ------------------------------------------
        Remove duplicate rows, both exact and key-column based, with full logging and traceability.
    
        Args:
            data (pd.DataFrame): DataFrame to deduplicate
    
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        import pandas as pd
        data_copy = data.copy()
        initial_count = len(data_copy)
    
        try:
            # ---------- Step 1: Remove exact duplicates ----------
            data_copy = data_copy.drop_duplicates()
            self.logger.debug(f"✅ Exact duplicates removed. Rows remaining: {len(data_copy)}")
    
            # ---------- Step 2: Remove duplicates based on key columns ----------
            key_columns = [col for col in ['State', 'Year', 'Vehicle Class'] if col in data_copy.columns]
            if key_columns:
                data_copy = data_copy.drop_duplicates(subset=key_columns, keep='first')
                self.logger.debug(f"✅ Duplicates removed based on key columns {key_columns}. Rows remaining: {len(data_copy)}")
            else:
                self.logger.debug("⚠️ No key columns found for subset deduplication")
    
            # ---------- Step 3: Summary ----------
            removed_count = initial_count - len(data_copy)
            if removed_count > 0:
                self.logger.info(f"🧹 Total duplicates removed: {removed_count}")
            else:
                self.logger.debug("No duplicates found to remove")
    
            return data_copy
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error during duplicate removal: {e}")
            return data_copy  # Return the copy even if partial failure occurs
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MAXED EDITION (2025 Hardened Missing Value Handler)
        ---------------------------------------------------
        Fill missing values in numeric, text, and categorical columns with adaptive defaults.
    
        Args:
            data (pd.DataFrame): DataFrame to process
    
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        import numpy as np
        data_copy = data.copy()
    
        try:
            # ---------- Step 1: Numeric columns ----------
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_copy[numeric_cols] = data_copy[numeric_cols].fillna(0)
                self.logger.debug(f"✅ Numeric columns filled with 0: {list(numeric_cols)}")
    
            # ---------- Step 2: Text / Object columns ----------
            text_cols = data_copy.select_dtypes(include=['object']).columns
            for col in text_cols:
                default_value = 'Unknown'
                if 'state' in col.lower():
                    default_value = 'Unknown State'
                elif 'vehicle' in col.lower():
                    default_value = 'Unknown Vehicle'
                elif 'city' in col.lower() or 'district' in col.lower():
                    default_value = 'Unknown Location'
    
                data_copy[col] = data_copy[col].fillna(default_value)
                self.logger.debug(f"✅ Text column '{col}' missing values filled with '{default_value}'")
    
            # ---------- Step 3: Summary ----------
            self.logger.info(f"✅ Missing values handled. Shape: {data_copy.shape}")
            return data_copy
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error handling missing values: {e}")
            return data_copy  # Return partially filled copy even on failure
    
    def clean_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MAXED EDITION (2025 Hardened Full Cleaning Pipeline)
        ---------------------------------------------------
        Perform end-to-end data cleaning: validation, numeric/text cleaning, 
        temporal extraction, vehicle categorization, deduplication, and missing value handling.
    
        Args:
            data (pd.DataFrame): Raw DataFrame to clean
    
        Returns:
            pd.DataFrame: Fully cleaned DataFrame
    
        Raises:
            DataProcessingError: If cleaning fails
        """
        try:
            self.logger.info("🧹 Starting comprehensive data cleaning...")
    
            # ---------- Step 1: Validate input data ----------
            self.validate_data(data)
    
            # ---------- Step 2: Remove completely empty rows ----------
            cleaned_data = data.dropna(how='all')
            self.logger.debug(f"Removed completely empty rows. Remaining rows: {len(cleaned_data)}")
    
            # ---------- Step 3: Clean numeric columns ----------
            cleaned_data = self.clean_numeric_columns(cleaned_data)
    
            # ---------- Step 4: Clean text columns ----------
            cleaned_data = self.clean_text_columns(cleaned_data)
    
            # ---------- Step 5: Extract temporal data ----------
            cleaned_data = self.extract_temporal_data(cleaned_data)
    
            # ---------- Step 6: Vehicle categorization ----------
            if 'Vehicle Class' in cleaned_data.columns:
                cleaned_data['Vehicle_Category'] = cleaned_data['Vehicle Class']
                self.logger.info("✅ Using actual Vehicle Class names as categories")
            else:
                cleaned_data['Vehicle_Category'] = cleaned_data.apply(self.categorize_vehicle, axis=1)
                self.logger.info("⚠️ Using generic vehicle categories as fallback")
    
            # ---------- Step 7: Remove duplicates ----------
            cleaned_data = self.remove_duplicates(cleaned_data)
    
            # ---------- Step 8: Handle missing values ----------
            cleaned_data = self.handle_missing_values(cleaned_data)
    
            self.logger.info(f"✅ Comprehensive data cleaning completed. Final shape: {cleaned_data.shape}")
            return cleaned_data
    
        except Exception as e:
            self.logger.error(f"❌ Data cleaning failed: {e}")
            raise DataProcessingError(f"Data cleaning failed: {e}")
