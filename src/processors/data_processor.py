"""
Main data processing module for VAHAN vehicle registration data.
Handles data processing, growth metrics calculation, and insights generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time

from .data_cleaner import DataCleaner
from ..core.config import Config
from ..core.exceptions import DataProcessingError
from ..core.models import GrowthMetrics, MarketInsights, ProcessingResult
from ..utils.logging_utils import get_logger
from ..utils.data_utils import create_sample_data

class VahanDataProcessor:
    """Process and analyze VAHAN vehicle registration data with investor-focused metrics."""
    
    def __init__(self):
        """MAXED EDITION: Initialize the DataProcessor with logging, cleaner, and placeholders."""
        from logging import getLogger, StreamHandler, Formatter, DEBUG
        from typing import Optional, Dict
        import pandas as pd
    
        # ---------- Data placeholders ----------
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.growth_metrics: Dict = {}
    
        # ---------- Initialize logger ----------
        self.logger = getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():  # Prevent duplicate handlers
            handler = StreamHandler()
            formatter = Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(DEBUG)  # Maxed: full debug trace
        self.logger.debug("✅ DataProcessor initialized with debug logger")
    
        # ---------- Initialize DataCleaner ----------
        self.cleaner = DataCleaner()
        self.logger.debug("✅ DataCleaner instance created and ready")
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        MAXED EDITION (2025 Hardened Loader)
        ------------------------------------
        Load data from CSV or other delimited file safely with validation:
        ✓ File existence check
        ✓ Empty/malformed file handling
        ✓ NaN normalization
        ✓ Full debug & info logging
        ✓ Returns a DataFrame ready for cleaning and processing
    
        Args:
            file_path (Union[str, Path]): Path to the CSV file
    
        Returns:
            pd.DataFrame: Loaded data
    
        Raises:
            FileNotFoundError: If the file does not exist
            DataProcessingError: If file is empty or cannot be read
        """
        from pathlib import Path
        import pandas as pd
    
        file_path = Path(file_path)
    
        # ---------- Step 1: File existence ----------
        if not file_path.exists() or not file_path.is_file():
            self.logger.error(f"❌ File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
    
        try:
            # ---------- Step 2: Load CSV safely ----------
            self.data = pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=True,
                na_values=['', 'NA', 'nan', 'None', '-'],
                encoding='utf-8'
            )
    
            # ---------- Step 3: Validate loaded data ----------
            if self.data.empty:
                self.logger.error(f"❌ Loaded file is empty: {file_path}")
                raise DataProcessingError("Loaded file is empty")
    
            self.logger.info(
                f"✅ Loaded {len(self.data)} rows x {len(self.data.columns)} columns from {file_path}"
            )
    
            # ---------- Step 4: Quick preview log ----------
            preview = self.data.head(3).to_dict(orient='records')
            self.logger.debug(f"📄 Data preview (first 3 rows): {preview}")
    
            return self.data
    
        except pd.errors.EmptyDataError:
            self.logger.error(f"❌ CSV file is empty or malformed: {file_path}")
            raise DataProcessingError(f"CSV file is empty or malformed: {file_path}")
        except pd.errors.ParserError as e:
            self.logger.error(f"❌ CSV parsing failed: {e}")
            raise DataProcessingError(f"CSV parsing failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error loading data from {file_path}: {e}")
            raise DataProcessingError(f"Unexpected error loading data: {e}")
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        MAXED EDITION (2025 Hardened Cleaner)
        -------------------------------------
        Clean, standardize, and enrich VAHAN data using the DataCleaner:
        ✓ Full numeric/text normalization
        ✓ Temporal extraction (Year/State)
        ✓ Vehicle categorization
        ✓ Deduplication & missing value handling
        ✓ Logs detailed debug/info messages
        ✓ Returns a boardroom-ready cleaned DataFrame
    
        Args:
            data (Optional[pd.DataFrame]): DataFrame to clean (uses self.data if None)
    
        Returns:
            pd.DataFrame: Cleaned and fully processed DataFrame
    
        Raises:
            DataProcessingError: If no data is available or cleaning fails
        """
        import pandas as pd
    
        # ---------- Step 1: Determine source of data ----------
        if data is None:
            if self.data is None:
                self.logger.error("❌ No data available to clean. Load data first.")
                raise DataProcessingError("No data available to clean. Load data first.")
            data_to_clean = self.data.copy()
        else:
            data_to_clean = data.copy()
    
        # ---------- Step 2: Handle empty DataFrame ----------
        if data_to_clean.empty:
            self.logger.warning("⚠️ DataFrame is empty. Nothing to clean.")
            return data_to_clean
    
        # ---------- Step 3: Perform MAXED cleaning ----------
        try:
            self.logger.info("🧹 Starting MAXED data cleaning via DataCleaner...")
    
            cleaned_data = self.cleaner.clean_all(data_to_clean)
    
            # ---------- Step 4: Quick post-cleaning validation ----------
            if cleaned_data.empty:
                self.logger.warning("⚠️ Cleaning produced an empty DataFrame.")
            else:
                self.logger.debug(
                    f"Cleaned data preview (first 3 rows): {cleaned_data.head(3).to_dict(orient='records')}"
                )
    
            self.processed_data = cleaned_data
            self.logger.info(f"✅ Data cleaning completed. Final shape: {cleaned_data.shape}")
    
            return cleaned_data
    
        except Exception as e:
            self.logger.error(f"❌ MAXED data cleaning failed: {e}")
            raise DataProcessingError(f"MAXED data cleaning failed: {e}")
    
    def calculate_growth_metrics(self, data: pd.DataFrame = None) -> Dict:
        """
        MAXED EDITION (2025 Hardened Growth Engine)
        ------------------------------------------
        Calculate YoY and QoQ growth rates for vehicle categories, states, 
        and manufacturers using processed or provided data.
        ✓ Noise-tolerant
        ✓ Missing-data aware
        ✓ Logs detailed debug/info
        ✓ Returns structured, boardroom-ready growth metrics
    
        Args:
            data (pd.DataFrame, optional): DataFrame to analyze (defaults to self.processed_data)
    
        Returns:
            Dict: Growth metrics including YoY and QoQ rates
        """
        from typing import Dict
    
        # ---------- Step 1: Determine source of data ----------
        if data is None:
            data = self.processed_data
    
        # ---------- Step 2: Validate data ----------
        if data is None or data.empty:
            self.logger.warning("⚠️ No data available for growth calculation")
            return {}
    
        try:
            self.logger.info("📈 Starting MAXED growth metrics calculation...")
    
            # ---------- Step 3: Initialize metrics dictionary ----------
            growth_metrics: Dict = {
                'yoy_growth': {},
                'qoq_growth': {},
                'category_growth': {},
                'state_growth': {},
                'manufacturer_growth': {}
            }
    
            # ---------- Step 4: Calculate total YoY growth ----------
            if 'TOTAL' in data.columns and 'Year' in data.columns:
                growth_metrics['yoy_growth'] = self._calculate_yoy_growth(data, 'TOTAL')
                self.logger.debug("✅ YoY growth for TOTAL calculated")
    
            # ---------- Step 5: YoY growth by category ----------
            growth_metrics['category_growth'] = self._calculate_yoy_by_category(data)
            self.logger.debug("✅ YoY growth by category calculated")
    
            # ---------- Step 6: YoY growth by state ----------
            growth_metrics['state_growth'] = self._calculate_yoy_by_state(data)
            self.logger.debug("✅ YoY growth by state calculated")
    
            # ---------- Step 7: Manufacturer trends ----------
            growth_metrics['manufacturer_growth'] = self._analyze_manufacturer_trends(data)
            self.logger.debug("✅ Manufacturer trends analyzed")
    
            # ---------- Step 8: Optional: QoQ growth (if available) ----------
            if 'Quarter' in data.columns:
                growth_metrics['qoq_growth'] = self._calculate_qoq_growth(data)
                self.logger.debug("✅ QoQ growth calculated")
    
            # ---------- Step 9: Store and return metrics ----------
            self.growth_metrics = growth_metrics
            self.logger.info("✅ MAXED growth metrics calculation completed")
    
            return growth_metrics
    
        except Exception as e:
            self.logger.error(f"❌ MAXED growth metrics calculation failed: {e}")
            return {}
    
    def _calculate_yoy_growth(self, data: pd.DataFrame, column: str) -> Dict:
        """
        MAXED EDITION (2025 Hardened YoY Engine)
        ----------------------------------------
        Calculate year-over-year (YoY) growth for a specific numeric column.
        ✓ Handles missing, zero, or noisy data
        ✓ Returns boardroom-ready structured dictionary
        ✓ Logs detailed debug and warnings
    
        Args:
            data (pd.DataFrame): DataFrame containing 'Year' and target column
            column (str): Column name for which to calculate YoY growth
    
        Returns:
            Dict: Mapping of "previous_year-current_year" to growth percentage
        """
        from typing import Dict
    
        growth_rates: Dict = {}
    
        try:
            # ---------- Step 1: Validate required columns ----------
            if 'Year' not in data.columns or column not in data.columns:
                self.logger.warning(f"⚠️ Required columns missing for YoY calculation: 'Year' or '{column}'")
                return growth_rates
    
            # ---------- Step 2: Ensure Year is numeric ----------
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data = data.dropna(subset=['Year', column])
    
            if data.empty:
                self.logger.warning(f"⚠️ No valid data available for YoY calculation on column '{column}'")
                return growth_rates
    
            # ---------- Step 3: Aggregate by year ----------
            yearly_data = data.groupby('Year', as_index=False)[column].sum()
            yearly_data = yearly_data.sort_values('Year')
    
            # ---------- Step 4: Calculate YoY growth ----------
            for i in range(1, len(yearly_data)):
                prev_year = int(yearly_data.iloc[i - 1]['Year'])
                curr_year = int(yearly_data.iloc[i]['Year'])
                prev_value = yearly_data.iloc[i - 1][column]
                curr_value = yearly_data.iloc[i][column]
    
                if prev_value > 0:
                    growth_rate = ((curr_value - prev_value) / prev_value) * 100
                    growth_rates[f"{prev_year}-{curr_year}"] = round(growth_rate, 2)
                else:
                    growth_rates[f"{prev_year}-{curr_year}"] = None
                    self.logger.debug(f"⚠️ Previous year value is zero for {prev_year}, cannot calculate growth")
    
            self.logger.debug(f"✅ YoY growth calculated for column '{column}' with {len(growth_rates)} entries")
            return growth_rates
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating YoY growth for '{column}': {e}")
            return growth_rates
    
    def _calculate_yoy_by_category(self, data: pd.DataFrame) -> Dict:
        """
        MAXED EDITION (2025 Hardened Category YoY Engine)
        -------------------------------------------------
        Calculate year-over-year (YoY) growth for each vehicle category.
        ✓ Handles missing, zero, or noisy data
        ✓ Noise-tolerant and missing-data aware
        ✓ Logs detailed warnings and debug info
        ✓ Returns structured dictionary for dashboards
    
        Args:
            data (pd.DataFrame): Must contain 'Vehicle_Category', 'Year', 'TOTAL'
    
        Returns:
            Dict: Mapping of category -> YoY growth dictionary
        """
        from typing import Dict
        import pandas as pd
    
        category_growth: Dict = {}
    
        try:
            # ---------- Step 1: Validate required columns ----------
            required_cols = ['Vehicle_Category', 'Year', 'TOTAL']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing required columns for category growth: {missing_cols}")
                return category_growth
    
            # ---------- Step 2: Ensure numeric Year and TOTAL ----------
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce')
            data = data.dropna(subset=['Year', 'Vehicle_Category', 'TOTAL'])
    
            if data.empty:
                self.logger.warning("⚠️ No valid data available for category YoY calculation")
                return category_growth
    
            # ---------- Step 3: Iterate through each category ----------
            for category in data['Vehicle_Category'].dropna().unique():
                cat_data = data[data['Vehicle_Category'] == category].copy()
                cat_yearly = cat_data.groupby('Year', as_index=False)['TOTAL'].sum()
                cat_yearly = cat_yearly.sort_values('Year')
    
                growth_rates: Dict = {}
                for i in range(1, len(cat_yearly)):
                    prev_year = int(cat_yearly.iloc[i - 1]['Year'])
                    curr_year = int(cat_yearly.iloc[i]['Year'])
                    prev_value = cat_yearly.iloc[i - 1]['TOTAL']
                    curr_value = cat_yearly.iloc[i]['TOTAL']
    
                    if prev_value > 0:
                        growth = ((curr_value - prev_value) / prev_value) * 100
                        growth_rates[f"{prev_year}-{curr_year}"] = round(growth, 2)
                    else:
                        growth_rates[f"{prev_year}-{curr_year}"] = None
                        self.logger.debug(f"⚠️ Previous year TOTAL is zero for {category} ({prev_year}), cannot calculate growth")
    
                category_growth[category] = growth_rates
    
            self.logger.debug(f"✅ YoY growth calculated for {len(category_growth)} vehicle categories")
            return category_growth
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating YoY growth by category: {e}")
            return {}
    
    def _calculate_yoy_by_state(self, data: pd.DataFrame) -> Dict:
        """
        MAXED EDITION (2025 Hardened State YoY Engine)
        ----------------------------------------------
        Calculate year-over-year (YoY) growth for each state.
        ✓ Handles missing, zero, or noisy data
        ✓ Noise-tolerant and missing-data aware
        ✓ Logs detailed warnings and debug info
        ✓ Returns structured dictionary for dashboards
    
        Args:
            data (pd.DataFrame): Must contain 'State', 'Year', 'TOTAL'
    
        Returns:
            Dict: Mapping of state -> YoY growth dictionary
        """
        from typing import Dict
        import pandas as pd
    
        state_growth: Dict = {}
    
        try:
            # ---------- Step 1: Validate required columns ----------
            required_cols = ['State', 'Year', 'TOTAL']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing required columns for state growth: {missing_cols}")
                return state_growth
    
            # ---------- Step 2: Ensure numeric Year and TOTAL ----------
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce')
            data = data.dropna(subset=['Year', 'State', 'TOTAL'])
    
            if data.empty:
                self.logger.warning("⚠️ No valid data available for state YoY calculation")
                return state_growth
    
            # ---------- Step 3: Iterate through each state ----------
            for state in data['State'].dropna().unique():
                state_data = data[data['State'] == state].copy()
                yearly_totals = state_data.groupby('Year', as_index=False)['TOTAL'].sum()
                yearly_totals = yearly_totals.sort_values('Year')
    
                growth_rates: Dict = {}
                for i in range(1, len(yearly_totals)):
                    prev_year = int(yearly_totals.iloc[i - 1]['Year'])
                    curr_year = int(yearly_totals.iloc[i]['Year'])
                    prev_value = yearly_totals.iloc[i - 1]['TOTAL']
                    curr_value = yearly_totals.iloc[i]['TOTAL']
    
                    if prev_value > 0:
                        growth = ((curr_value - prev_value) / prev_value) * 100
                        growth_rates[f"{prev_year}-{curr_year}"] = round(growth, 2)
                    else:
                        growth_rates[f"{prev_year}-{curr_year}"] = None
                        self.logger.debug(f"⚠️ Previous year TOTAL is zero for {state} ({prev_year}), cannot calculate growth")
    
                state_growth[state] = growth_rates
    
            self.logger.debug(f"✅ YoY growth calculated for {len(state_growth)} states")
            return state_growth
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating YoY growth by state: {e}")
            return {}
    
    def _analyze_manufacturer_trends(self, data: pd.DataFrame) -> Dict:
        """
        MAXED EDITION (2025 Hardened Manufacturer Trend Engine)
        -------------------------------------------------------
        Analyze manufacturer and vehicle class trends:
        ✓ Top manufacturers
        ✓ Growth rates over years
        ✓ Market share calculations
        ✓ Handles missing, noisy, or zero data gracefully
        ✓ Detailed logging for dashboards and automated reporting
    
        Args:
            data (pd.DataFrame): Must contain 'Vehicle Class' and optionally 'Manufacturer', 'Year', 'TOTAL'
    
        Returns:
            Dict: Structured dictionary with top manufacturers, growth rates, and market share
        """
        from typing import Dict
        import pandas as pd
        import numpy as np
    
        manufacturer_data: Dict = {}
    
        try:
            # ---------- Step 0: Validate required columns ----------
            if 'Vehicle Class' not in data.columns:
                self.logger.warning("⚠️ 'Vehicle Class' column missing. Cannot analyze manufacturer trends.")
                return manufacturer_data
    
            # Optional columns
            if 'Manufacturer' not in data.columns:
                data['Manufacturer'] = 'Unknown Manufacturer'
                self.logger.warning("⚠️ 'Manufacturer' column missing. Using placeholder values.")
    
            if 'TOTAL' not in data.columns:
                data['TOTAL'] = 0
                self.logger.warning("⚠️ 'TOTAL' column missing. Defaulting to 0.")
    
            # Coerce numeric columns
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)
            if 'Year' in data.columns:
                data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            else:
                data['Year'] = np.nan
    
            # ---------- Step 1: Calculate top manufacturers ----------
            top_manufacturers_df = data.groupby('Manufacturer', as_index=False)['TOTAL'].sum()
            top_manufacturers_df = top_manufacturers_df.sort_values('TOTAL', ascending=False).head(10)
            manufacturer_data['top_manufacturers'] = top_manufacturers_df.to_dict(orient='records')
            self.logger.debug(f"✅ Top manufacturers determined: {manufacturer_data['top_manufacturers']}")
    
            # ---------- Step 2: Calculate manufacturer growth rates ----------
            growth_rates: Dict = {}
            for manufacturer in data['Manufacturer'].dropna().unique():
                man_data = data[data['Manufacturer'] == manufacturer]
                if 'Year' not in man_data.columns or man_data['Year'].isna().all():
                    growth_rates[manufacturer] = {}
                    continue
    
                yearly_totals = man_data.groupby('Year', as_index=False)['TOTAL'].sum().sort_values('Year')
                man_growth: Dict = {}
                for i in range(1, len(yearly_totals)):
                    prev_year = int(yearly_totals.iloc[i - 1]['Year'])
                    curr_year = int(yearly_totals.iloc[i]['Year'])
                    prev_value = yearly_totals.iloc[i - 1]['TOTAL']
                    curr_value = yearly_totals.iloc[i]['TOTAL']
    
                    if prev_value > 0:
                        growth = ((curr_value - prev_value) / prev_value) * 100
                        man_growth[f"{prev_year}-{curr_year}"] = round(growth, 2)
                    else:
                        man_growth[f"{prev_year}-{curr_year}"] = None
                        self.logger.debug(f"⚠️ Previous year TOTAL is zero for {manufacturer} ({prev_year}), cannot calculate growth")
    
                growth_rates[manufacturer] = man_growth
    
            manufacturer_data['growth_rates'] = growth_rates
            self.logger.debug("✅ Manufacturer growth rates calculated")
    
            # ---------- Step 3: Calculate market share ----------
            total_registrations = data['TOTAL'].sum()
            if total_registrations > 0:
                market_share = (data.groupby('Manufacturer')['TOTAL'].sum() / total_registrations * 100).round(2).to_dict()
            else:
                market_share = {man: 0 for man in data['Manufacturer'].unique()}
    
            manufacturer_data['market_share'] = market_share
            self.logger.debug(f"✅ Manufacturer market share calculated: {market_share}")
    
            self.logger.info("✅ Manufacturer trend analysis fully completed")
            return manufacturer_data
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error analyzing manufacturer trends: {e}")
            return {}
    
    def _get_top_manufacturers(self, data: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """
        MAXED 2025 EDITION
        ------------------
        Determine top manufacturers by total vehicle registrations.
    
        Handles:
            - Missing 'Vehicle Class' or 'TOTAL' columns
            - Non-numeric 'TOTAL' values
            - Empty or malformed datasets
            - Dynamic top_n limits
    
        Args:
            data (pd.DataFrame): Must contain 'Vehicle Class' and 'TOTAL' columns
            top_n (int): Number of top manufacturers to return (default 10)
    
        Returns:
            list[dict]: Each dict contains 'Vehicle Class' and summed 'TOTAL' registrations
        """
        import pandas as pd
    
        try:
            # Step 0: Validate required columns
            if 'Vehicle Class' not in data.columns or 'TOTAL' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for top manufacturer calculation")
                return []
    
            # Step 1: Ensure TOTAL is numeric and safe
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)
    
            # Step 2: Group by Vehicle Class and sum TOTAL registrations
            manufacturer_totals = data.groupby('Vehicle Class', as_index=False)['TOTAL'].sum()
    
            if manufacturer_totals.empty:
                self.logger.warning("⚠️ No data available after grouping by 'Vehicle Class'")
                return []
    
            # Step 3: Sort descending and pick top_n
            top_manufacturers = manufacturer_totals.sort_values('TOTAL', ascending=False).head(top_n)
    
            self.logger.info(f"✅ Top {len(top_manufacturers)} manufacturers successfully calculated")
            return top_manufacturers.to_dict('records')
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating top manufacturers: {e}")
            return []
    
    def _get_manufacturer_growth(self, data: pd.DataFrame) -> dict:
        """
        MAXED 2025 EDITION
        ------------------
        Calculate year-over-year (YoY) growth rates for each manufacturer (Vehicle Class).
    
        Features:
            - Handles missing or non-numeric 'Year' or 'TOTAL'
            - Drops incomplete rows safely
            - Works even if only one year of data exists
            - Logs progress and warnings comprehensively
    
        Args:
            data (pd.DataFrame): Must contain 'Vehicle Class', 'Year', and 'TOTAL' columns
    
        Returns:
            dict: Mapping {manufacturer: latest YoY growth %, or None if cannot calculate}
        """
        from typing import Dict
        import pandas as pd
    
        manufacturer_growth: Dict = {}
    
        try:
            # Step 0: Validate required columns
            if 'Vehicle Class' not in data.columns or 'Year' not in data.columns or 'TOTAL' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for manufacturer growth calculation")
                return manufacturer_growth
    
            # Step 1: Ensure numeric values
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)
            data = data.dropna(subset=['Year', 'Vehicle Class'])
    
            if data.empty:
                self.logger.warning("⚠️ No valid data available for manufacturer growth calculation")
                return manufacturer_growth
    
            # Step 2: Iterate over unique manufacturers
            for manufacturer in data['Vehicle Class'].dropna().unique():
                mfg_data = data[data['Vehicle Class'] == manufacturer]
    
                if len(mfg_data) < 2:
                    self.logger.debug(f"⚠️ Not enough data to calculate growth for '{manufacturer}'")
                    continue
    
                # Group by Year and sum TOTAL
                yearly_totals = mfg_data.groupby('Year', as_index=False)['TOTAL'].sum().sort_values('Year')
    
                # Calculate YoY growth for the latest year only
                prev_val = yearly_totals.iloc[-2]['TOTAL']
                latest_val = yearly_totals.iloc[-1]['TOTAL']
    
                if prev_val > 0:
                    growth_rate = ((latest_val - prev_val) / prev_val) * 100
                    manufacturer_growth[manufacturer] = round(growth_rate, 2)
                else:
                    manufacturer_growth[manufacturer] = None
                    self.logger.debug(f"⚠️ Previous year TOTAL is zero for '{manufacturer}', cannot compute growth")
    
            self.logger.info(f"✅ Manufacturer growth calculated for {len(manufacturer_growth)} manufacturers")
            return manufacturer_growth
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating manufacturer growth: {e}")
            return {}
    
    def _calculate_market_share(self, data: pd.DataFrame) -> dict[str, float]:
        """
        MAXED 2025 EDITION
        ------------------
        Calculate market share (%) by manufacturer (Vehicle Class).
    
        Features:
            - Handles missing or non-numeric 'TOTAL' values
            - Returns empty dict if total registrations are zero
            - Logs debug and warning messages for robust traceability
    
        Args:
            data (pd.DataFrame): Must contain 'Vehicle Class' and 'TOTAL' columns
    
        Returns:
            dict[str, float]: Mapping {manufacturer: market share %}
        """
        from typing import Dict
        import pandas as pd
    
        try:
            # Step 0: Validate required columns
            if 'Vehicle Class' not in data.columns or 'TOTAL' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for market share calculation")
                return {}
    
            # Step 1: Ensure numeric TOTAL values
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)
    
            total_registrations = data['TOTAL'].sum()
            if total_registrations <= 0:
                self.logger.warning("⚠️ Total registrations are zero or negative. Cannot calculate market share.")
                return {}
    
            # Step 2: Aggregate by manufacturer
            manufacturer_totals = data.groupby('Vehicle Class', as_index=False)['TOTAL'].sum()
            manufacturer_totals['market_share'] = (manufacturer_totals['TOTAL'] / total_registrations * 100).round(2)
    
            self.logger.info(f"✅ Market share calculated for {len(manufacturer_totals)} manufacturers")
            return dict(zip(manufacturer_totals['Vehicle Class'], manufacturer_totals['market_share']))
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating market share: {e}")
            return {}
    
    def get_investor_insights(self, data: pd.DataFrame = None) -> dict:
        """
        MAXED 2025 EDITION
        ------------------
        Generate key investor insights from the processed VAHAN data.
    
        Features:
            - Uses processed data if no DataFrame is provided
            - Validates input data availability
            - Robust try/except logging for errors
            - Returns structured dict with overview, leaders, risks, and opportunities
    
        Args:
            data (pd.DataFrame, optional): DataFrame to analyze (defaults to self.processed_data)
    
        Returns:
            dict: Insights including market overview, growth leaders, risk factors, and opportunities
        """
        from typing import Dict
    
        # Step 0: Determine source of data
        if data is None:
            data = self.processed_data
    
        # Step 1: Validate data
        if data is None or data.empty:
            self.logger.warning("⚠️ No data available to generate investor insights")
            return {}
    
        try:
            self.logger.info("📊 Generating investor insights...")
    
            # Step 2: Assemble insights dictionary
            insights: Dict = {
                'market_overview': self._get_market_overview(data),  # MAXED: handles empty/malformed data
                'growth_leaders': self._get_growth_leaders(),        # MAXED: ensures valid lists
                'risk_factors': self._identify_risk_factors(),       # MAXED: fallback to empty list if needed
                'investment_opportunities': self._identify_opportunities()  # MAXED: fallback safe defaults
            }
    
            # Step 3: Validate output structure
            if not all(isinstance(v, (dict, list)) for v in insights.values()):
                self.logger.warning("⚠️ Some investor insights are missing or invalid")
            
            self.logger.info("✅ Investor insights generated successfully")
            return insights
    
        except Exception as e:
            self.logger.error(f"❌ Error generating investor insights: {e}", exc_info=True)
            return {}
    
    def _get_market_overview(self, data: pd.DataFrame) -> dict:
        """
        MAXED 2025 EDITION
        ------------------
        Generate a comprehensive market overview from VAHAN data.
    
        Features:
            - Total registrations
            - Data period (earliest and latest year)
            - Number of states covered
            - Vehicle category breakdown
            - Fully safe against missing/malformed columns
            - Detailed debug logging
        """
        from typing import Dict
        import pandas as pd
    
        overview: Dict = {}
    
        try:
            if data is None or data.empty:
                self.logger.warning("⚠️ Empty DataFrame provided for market overview")
                return overview
    
            # --- Total registrations ---
            if 'TOTAL' in data.columns:
                total_regs = pd.to_numeric(data['TOTAL'], errors='coerce').sum()
                overview['total_registrations'] = int(total_regs) if not pd.isna(total_regs) else 0
    
            # --- Data period (min-max year) ---
            if 'Year' in data.columns:
                years = pd.to_numeric(data['Year'], errors='coerce').dropna()
                if len(years) > 0:
                    overview['data_period'] = f"{int(years.min())}-{int(years.max())}"
    
            # --- States covered ---
            if 'State' in data.columns:
                overview['states_covered'] = int(data['State'].nunique())
    
            # --- Vehicle category breakdown ---
            if 'Vehicle_Category' in data.columns and 'TOTAL' in data.columns:
                cat_totals = (
                    data.groupby('Vehicle_Category', as_index=False)['TOTAL']
                    .sum()
                )
                overview['category_breakdown'] = {
                    str(row['Vehicle_Category']): int(row['TOTAL']) 
                    for _, row in cat_totals.iterrows()
                }
    
            self.logger.debug(f"✅ Market overview generated: {overview}")
            return overview
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating market overview: {e}", exc_info=True)
            return {}
    
    def _get_growth_leaders(self) -> list[str]:
        """
        MAXED 2025 EDITION
        ------------------
        Identify top growth leaders from pre-calculated growth metrics.
        Features:
            - Handles missing or empty metrics
            - Considers both category and state growth
            - Applies configurable growth thresholds
            - Logs detailed debug info
            - Returns top 5 leaders by growth percentage
        """
        from typing import List
    
        try:
            leaders: List[str] = []
    
            # --- Fetch category growth safely ---
            category_growth = self.growth_metrics.get('category_growth', {})
            for category, growth_data in category_growth.items():
                if isinstance(growth_data, dict) and growth_data:
                    # Latest YoY growth
                    latest_growth = next(reversed(list(growth_data.values())), 0)
                    if latest_growth is not None and latest_growth > 10:  # MAXED threshold
                        leaders.append(f"{category} category showing {latest_growth}% growth")
    
            # --- Fetch state growth safely ---
            state_growth = self.growth_metrics.get('state_growth', {})
            for state, growth_data in state_growth.items():
                if isinstance(growth_data, dict) and growth_data:
                    latest_growth = next(reversed(list(growth_data.values())), 0)
                    if latest_growth is not None and latest_growth > 15:  # MAXED threshold for states
                        leaders.append(f"{state} state showing {latest_growth}% growth")
    
            # Sort leaders by numeric growth descending
            def extract_growth(text: str) -> float:
                try:
                    return float(text.split()[-2].replace('%',''))
                except Exception:
                    return 0
    
            leaders_sorted = sorted(leaders, key=extract_growth, reverse=True)
    
            self.logger.debug(f"✅ Growth leaders identified: {leaders_sorted[:5]}")
            return leaders_sorted[:5]
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying growth leaders: {e}", exc_info=True)
            return []
    
    def _identify_risk_factors(self) -> list[str]:
        """
        MAXED 2025 EDITION
        ------------------
        Identify potential risk factors based on negative growth trends.
        Features:
            - Handles missing or empty growth metrics
            - Safely extracts latest YoY growth
            - Applies configurable negative growth threshold
            - Logs debug info
            - Returns top risk categories sorted by magnitude of decline
        """
        from typing import List
    
        try:
            risks: List[str] = []
    
            # --- Fetch category growth safely ---
            category_growth = self.growth_metrics.get('category_growth', {})
            for category, growth_data in category_growth.items():
                if isinstance(growth_data, dict) and growth_data:
                    # Latest YoY growth
                    latest_growth = next(reversed(list(growth_data.values())), 0)
                    if latest_growth is not None and latest_growth < -5:  # MAXED negative growth threshold
                        risks.append(f"{category} category declining by {abs(round(latest_growth, 2))}%")
    
            # Sort risks by magnitude descending
            def extract_decline(text: str) -> float:
                try:
                    return float(text.split()[-1].replace('%',''))
                except Exception:
                    return 0
    
            risks_sorted = sorted(risks, key=extract_decline, reverse=True)
    
            self.logger.debug(f"✅ Risk factors identified: {risks_sorted[:5]}")
            return risks_sorted[:5]
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying risk factors: {e}", exc_info=True)
            return []
    
    def _identify_opportunities(self) -> list[str]:
        """
        MAXED 2025 EDITION
        ------------------
        Identify top investment opportunities based on manufacturer growth trends.
        Features:
            - Handles missing or empty manufacturer growth data
            - Applies configurable high-growth threshold
            - Sorts opportunities by growth magnitude
            - Logs debug info with full context
            - Returns top 3 opportunities
        """
        from typing import List
    
        try:
            opportunities: List[str] = []
    
            # Fetch manufacturer growth safely
            manufacturer_growth = self.growth_metrics.get('manufacturer_growth', {})
            if isinstance(manufacturer_growth, dict):
                growth_rates = manufacturer_growth
            else:
                growth_rates = {}
    
            # Identify high-growth manufacturers
            for manufacturer, growth_rate in growth_rates.items():
                if growth_rate is not None and growth_rate > 20:  # MAXED high growth threshold
                    opportunities.append(f"{manufacturer} showing exceptional {round(growth_rate, 2)}% growth")
    
            # Sort by growth descending
            opportunities_sorted = sorted(
                opportunities,
                key=lambda x: float(x.split()[-2]),  # Extract numeric growth
                reverse=True
            )
    
            self.logger.debug(f"✅ Investment opportunities identified: {opportunities_sorted[:3]}")
            return opportunities_sorted[:3]
    
        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying opportunities: {e}", exc_info=True)
            return []
    
    def export_processed_data(self, filename: str = None) -> str:
        """
        MAXED 2025 EDITION
        ------------------
        Export processed VAHAN data and growth metrics to CSV files.
        Features:
            - Handles missing processed data safely
            - Generates Config-based filenames if not provided
            - Exports main data and nested growth metrics
            - Full logging and error handling
        """
        import pandas as pd
        from pathlib import Path
    
        if self.processed_data is None or self.processed_data.empty:
            self.logger.error("❌ No processed data available to export")
            raise DataProcessingError("No processed data available to export")
    
        # Generate filename if not provided
        if filename is None:
            filename = Config.get_output_filename("vahan_processed")
    
        filepath: Path = Config.OUTPUT_DIR / filename
        Config.ensure_directories()
    
        try:
            # Export main processed data
            self.processed_data.to_csv(filepath, index=False)
            self.logger.info(f"💾 Processed data exported to {filepath}")
    
            # Export growth metrics if available
            if self.growth_metrics:
                metrics_filename = filepath.with_name(f"metrics_{filepath.name}")
                all_metrics = []
    
                for metric_name, metric_data in self.growth_metrics.items():
                    if isinstance(metric_data, dict) and metric_data:
                        # Flatten nested dictionaries
                        for key, sub_data in metric_data.items():
                            if isinstance(sub_data, dict):
                                for sub_key, value in sub_data.items():
                                    all_metrics.append({
                                        'Metric_Type': metric_name,
                                        'Category': f"{key} - {sub_key}",
                                        'Value': value
                                    })
                            else:
                                all_metrics.append({
                                    'Metric_Type': metric_name,
                                    'Category': key,
                                    'Value': sub_data
                                })
    
                if all_metrics:
                    metrics_df = pd.DataFrame(all_metrics)
                    metrics_df.to_csv(metrics_filename, index=False)
                    self.logger.info(f"📊 Growth metrics exported to {metrics_filename}")
    
            return str(filepath)
    
        except Exception as e:
            self.logger.error(f"❌ Failed to export data: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to export data: {e}")
    
    def process_all(self, data: pd.DataFrame) -> "ProcessingResult":
        """
        MAXED 2025 EDITION
        ------------------
        Execute the full VAHAN data processing pipeline in one step.
    
        Steps:
            1. Clean the raw data (numeric/text, temporal extraction, deduplication, missing values)
            2. Calculate growth metrics (YoY, category, state, manufacturer)
            3. Generate investor insights (market overview, growth leaders, risks, opportunities)
            4. Return structured results with metadata and processing time
    
        Args:
            data (pd.DataFrame): Raw input VAHAN data
    
        Returns:
            ProcessingResult: Complete processing results including cleaned data, growth metrics,
                              insights, processing time, and record count
        """
        import time
    
        start_time = time.time()
    
        try:
            self.logger.info("🚀 Starting full VAHAN data processing pipeline...")
    
            # Step 1: Clean the data
            cleaned_data = self.clean_data(data)
            self.logger.info(f"🧹 Data cleaning completed: {len(cleaned_data)} records")
    
            # Step 2: Calculate growth metrics
            growth_metrics = self.calculate_growth_metrics(cleaned_data)
            self.logger.info("📈 Growth metrics calculation completed")
    
            # Step 3: Generate investor insights
            insights = self.get_investor_insights(cleaned_data)
            self.logger.info("📊 Investor insights generated")
    
            # Step 4: Measure processing time
            processing_time = time.time() - start_time
    
            # Step 5: Create structured result
            result = ProcessingResult(
                cleaned_data=cleaned_data,
                growth_metrics=GrowthMetrics(**growth_metrics) if growth_metrics else None,
                insights=MarketInsights(**insights) if insights else None,
                processing_time=processing_time,
                records_processed=len(cleaned_data)
            )
    
            self.logger.info(f"✅ Full VAHAN processing finished in {processing_time:.2f}s "
                             f"for {len(cleaned_data)} records")
            return result
    
        except Exception as e:
            self.logger.error(f"❌ Complete processing pipeline failed: {e}", exc_info=True)
            raise DataProcessingError(f"Complete processing pipeline failed: {e}")
