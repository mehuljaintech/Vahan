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
        """Initialize the DataProcessor with logging, cleaner, and placeholders for data."""
        from logging import getLogger, StreamHandler, Formatter, DEBUG
        from typing import Optional, Dict
        import pandas as pd

        # Data placeholders
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.growth_metrics: Dict = {}

        # Initialize logger
        self.logger = getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = StreamHandler()
            formatter = Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(DEBUG)
        self.logger.debug("DataProcessor initialized")

        # Initialize DataCleaner instance
        self.cleaner = DataCleaner()
        self.logger.debug("DataCleaner instance created")
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file with validation, ensuring file exists and is not empty.

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

        # --- Step 1: Check file existence ---
        if not file_path.exists() or not file_path.is_file():
            self.logger.error(f"❌ File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # --- Step 2: Load CSV with basic safety options ---
            self.data = pd.read_csv(file_path, dtype=str, keep_default_na=True, na_values=['', 'NA', 'nan'])
            
            # --- Step 3: Validate loaded data ---
            if self.data.empty:
                self.logger.error(f"❌ Loaded file is empty: {file_path}")
                raise DataProcessingError("Loaded file is empty")
            
            self.logger.info(f"✅ Loaded {len(self.data)} rows and {len(self.data.columns)} columns from {file_path}")
            return self.data

        except pd.errors.EmptyDataError:
            self.logger.error(f"❌ CSV file is empty or malformed: {file_path}")
            raise DataProcessingError(f"CSV file is empty or malformed: {file_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load data from {file_path}: {e}")
            raise DataProcessingError(f"Failed to load data: {e}")
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and standardize the VAHAN data using the DataCleaner.

        Args:
            data (Optional[pd.DataFrame]): DataFrame to clean (uses self.data if None)

        Returns:
            pd.DataFrame: Cleaned data

        Raises:
            DataProcessingError: If no data is available to clean
        """
        import pandas as pd

        # Step 1: Determine source of data
        if data is None:
            if self.data is None:
                self.logger.error("❌ No data available to clean. Load data first.")
                raise DataProcessingError("No data available to clean. Load data first.")
            data_to_clean = self.data.copy()
        else:
            data_to_clean = data.copy()

        # Step 2: Handle empty DataFrame
        if data_to_clean.empty:
            self.logger.warning("⚠️ DataFrame is empty. Nothing to clean.")
            return data_to_clean

        # Step 3: Perform comprehensive cleaning
        try:
            self.logger.info("🧹 Starting data cleaning via DataCleaner...")
            cleaned_data = self.cleaner.clean_all(data_to_clean)
            self.processed_data = cleaned_data
            self.logger.info(f"✅ Data cleaning completed. Final shape: {cleaned_data.shape}")
            return cleaned_data

        except Exception as e:
            self.logger.error(f"❌ Data cleaning failed: {e}")
            raise DataProcessingError(f"Data cleaning failed: {e}")
    
    def calculate_growth_metrics(self, data: pd.DataFrame = None) -> Dict:
        """
        Calculate YoY and QoQ growth rates for vehicle categories, states, 
        and manufacturers using processed or provided data.

        Args:
            data (pd.DataFrame, optional): DataFrame to analyze (defaults to self.processed_data)

        Returns:
            Dict: Growth metrics including YoY and QoQ rates
        """
        from typing import Dict

        # Step 1: Determine source of data
        if data is None:
            data = self.processed_data

        # Step 2: Validate data availability
        if data is None or data.empty:
            self.logger.warning("⚠️ No data available for growth calculation")
            return {}

        try:
            self.logger.info("📈 Starting growth metrics calculation...")

            # Initialize metrics dictionary
            growth_metrics: Dict = {
                'yoy_growth': {},
                'qoq_growth': {},
                'category_growth': {},
                'state_growth': {},
                'manufacturer_growth': {}
            }

            # Step 3: Calculate total YoY growth
            if 'TOTAL' in data.columns and 'Year' in data.columns:
                growth_metrics['yoy_growth'] = self._calculate_yoy_growth(data, 'TOTAL')
                self.logger.debug("✅ YoY growth for TOTAL calculated")

            # Step 4: Calculate YoY growth by category
            growth_metrics['category_growth'] = self._calculate_yoy_by_category(data)
            self.logger.debug("✅ YoY growth by category calculated")

            # Step 5: Calculate YoY growth by state
            growth_metrics['state_growth'] = self._calculate_yoy_by_state(data)
            self.logger.debug("✅ YoY growth by state calculated")

            # Step 6: Analyze manufacturer trends
            growth_metrics['manufacturer_growth'] = self._analyze_manufacturer_trends(data)
            self.logger.debug("✅ Manufacturer trends analyzed")

            # Step 7: Store and return metrics
            self.growth_metrics = growth_metrics
            self.logger.info("✅ Growth metrics calculation completed")
            return growth_metrics

        except Exception as e:
            self.logger.error(f"❌ Error calculating growth metrics: {e}")
            return {}
    
    def _calculate_yoy_growth(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculate year-over-year (YoY) growth for a specific numeric column.

        Args:
            data (pd.DataFrame): DataFrame containing 'Year' and target column
            column (str): Column name for which to calculate YoY growth

        Returns:
            Dict: Mapping of "previous_year-current_year" to growth percentage
        """
        from typing import Dict

        growth_rates: Dict = {}

        try:
            # Validate required columns
            if 'Year' not in data.columns or column not in data.columns:
                self.logger.warning(f"⚠️ Required columns missing for YoY calculation: 'Year' or '{column}'")
                return growth_rates

            # Ensure Year is numeric for proper sorting
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data = data.dropna(subset=['Year', column])

            # Group by year and sum the values
            yearly_data = data.groupby('Year', as_index=False)[column].sum()
            yearly_data = yearly_data.sort_values('Year')

            # Calculate YoY growth
            for i in range(1, len(yearly_data)):
                prev_year = yearly_data.iloc[i - 1]['Year']
                curr_year = yearly_data.iloc[i]['Year']
                prev_value = yearly_data.iloc[i - 1][column]
                curr_value = yearly_data.iloc[i][column]

                if prev_value > 0:
                    growth_rate = ((curr_value - prev_value) / prev_value) * 100
                    growth_rates[f"{int(prev_year)}-{int(curr_year)}"] = round(growth_rate, 2)
                else:
                    growth_rates[f"{int(prev_year)}-{int(curr_year)}"] = None  # Cannot divide by zero

            self.logger.debug(f"✅ YoY growth calculated for column '{column}'")
            return growth_rates

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating YoY growth for '{column}': {e}")
            return growth_rates
    
    def _calculate_yoy_by_category(self, data: pd.DataFrame) -> Dict:
        """
        Calculate year-over-year (YoY) growth for each vehicle category.

        Args:
            data (pd.DataFrame): DataFrame containing 'Vehicle_Category', 'Year', and 'TOTAL' columns

        Returns:
            Dict: Mapping of category to YoY growth rates
        """
        from typing import Dict
        import pandas as pd

        category_growth: Dict = {}

        try:
            # Validate required columns
            if 'Vehicle_Category' not in data.columns or 'Year' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for category growth calculation")
                return category_growth

            # Ensure Year column is numeric
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data = data.dropna(subset=['Year', 'Vehicle_Category'])

            # Iterate through unique categories
            for category in data['Vehicle_Category'].dropna().unique():
                category_data = data[data['Vehicle_Category'] == category]

                if 'TOTAL' not in category_data.columns:
                    continue

                # Group by Year and sum TOTAL
                yearly_totals = category_data.groupby('Year', as_index=False)['TOTAL'].sum()
                yearly_totals = yearly_totals.sort_values('Year')

                growth_rates: Dict = {}
                for i in range(1, len(yearly_totals)):
                    prev_year = yearly_totals.iloc[i - 1]['Year']
                    curr_year = yearly_totals.iloc[i]['Year']
                    prev_value = yearly_totals.iloc[i - 1]['TOTAL']
                    curr_value = yearly_totals.iloc[i]['TOTAL']

                    if prev_value > 0:
                        growth_rate = ((curr_value - prev_value) / prev_value) * 100
                        growth_rates[f"{int(prev_year)}-{int(curr_year)}"] = round(growth_rate, 2)
                    else:
                        growth_rates[f"{int(prev_year)}-{int(curr_year)}"] = None  # Cannot divide by zero

                category_growth[category] = growth_rates

            self.logger.debug("✅ YoY growth calculated for all vehicle categories")
            return category_growth

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating YoY growth by category: {e}")
            return {}
    
    def _calculate_yoy_by_state(self, data: pd.DataFrame) -> Dict:
        """
        Calculate year-over-year (YoY) growth for each state.

        Args:
            data (pd.DataFrame): DataFrame containing 'State', 'Year', and 'TOTAL' columns

        Returns:
            Dict: Mapping of state to YoY growth rates
        """
        from typing import Dict
        import pandas as pd

        state_growth: Dict = {}

        try:
            # Validate required columns
            if 'State' not in data.columns or 'Year' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for state growth calculation")
                return state_growth

            # Ensure Year column is numeric
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data = data.dropna(subset=['Year', 'State'])

            # Iterate through unique states
            for state in data['State'].dropna().unique():
                state_data = data[data['State'] == state]

                if 'TOTAL' not in state_data.columns:
                    continue

                # Group by Year and sum TOTAL
                yearly_totals = state_data.groupby('Year', as_index=False)['TOTAL'].sum()
                yearly_totals = yearly_totals.sort_values('Year')

                growth_rates: Dict = {}
                for i in range(1, len(yearly_totals)):
                    prev_year = yearly_totals.iloc[i - 1]['Year']
                    curr_year = yearly_totals.iloc[i]['Year']
                    prev_value = yearly_totals.iloc[i - 1]['TOTAL']
                    curr_value = yearly_totals.iloc[i]['TOTAL']

                    if prev_value > 0:
                        growth_rate = ((curr_value - prev_value) / prev_value) * 100
                        growth_rates[f"{int(prev_year)}-{int(curr_year)}"] = round(growth_rate, 2)
                    else:
                        growth_rates[f"{int(prev_year)}-{int(curr_year)}"] = None  # Cannot divide by zero

                state_growth[state] = growth_rates

            self.logger.debug("✅ YoY growth calculated for all states")
            return state_growth

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating YoY growth by state: {e}")
            return {}
    
    def _analyze_manufacturer_trends(self, data: pd.DataFrame) -> Dict:
        """
        Analyze manufacturer and vehicle class trends including top manufacturers,
        growth rates, and market share.

        Args:
            data (pd.DataFrame): DataFrame containing 'Vehicle Class' and related columns

        Returns:
            Dict: Manufacturer trends including top manufacturers, growth rates, and market share
        """
        from typing import Dict

        manufacturer_data: Dict = {}

        try:
            if 'Vehicle Class' not in data.columns:
                self.logger.warning("⚠️ 'Vehicle Class' column missing. Cannot analyze manufacturer trends.")
                return manufacturer_data

            self.logger.info("🔍 Analyzing manufacturer trends...")

            # Step 1: Get top manufacturers by total registrations
            top_manufacturers = self._get_top_manufacturers(data)
            manufacturer_data['top_manufacturers'] = top_manufacturers
            self.logger.debug(f"✅ Top manufacturers calculated: {top_manufacturers}")

            # Step 2: Calculate manufacturer growth rates
            manufacturer_growth = self._get_manufacturer_growth(data)
            manufacturer_data['growth_rates'] = manufacturer_growth
            self.logger.debug("✅ Manufacturer growth rates calculated")

            # Step 3: Calculate market share
            market_share = self._calculate_market_share(data)
            manufacturer_data['market_share'] = market_share
            self.logger.debug("✅ Manufacturer market share calculated")

            self.logger.info("✅ Manufacturer trend analysis completed")
            return manufacturer_data

        except Exception as e:
            self.logger.warning(f"⚠️ Error analyzing manufacturer trends: {e}")
            return {}
    
    def _get_top_manufacturers(self, data: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """
        Get top manufacturers (Vehicle Class) by total registrations.

        Args:
            data (pd.DataFrame): DataFrame containing 'Vehicle Class' and 'TOTAL' columns
            top_n (int): Number of top manufacturers to return

        Returns:
            list[dict]: List of dictionaries with top manufacturers and their total registrations
        """
        import pandas as pd

        try:
            # Validate required columns
            if 'Vehicle Class' not in data.columns or 'TOTAL' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for top manufacturer calculation")
                return []

            # Ensure TOTAL is numeric
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)

            # Group by Vehicle Class and sum TOTAL
            manufacturer_totals = data.groupby('Vehicle Class', as_index=False)['TOTAL'].sum()

            # Sort descending and take top_n
            top_manufacturers = manufacturer_totals.sort_values('TOTAL', ascending=False).head(top_n)

            self.logger.debug(f"✅ Top {top_n} manufacturers calculated")
            return top_manufacturers.to_dict('records')

        except Exception as e:
            self.logger.warning(f"⚠️ Error getting top manufacturers: {e}")
            return []
    
    def _get_manufacturer_growth(self, data: pd.DataFrame) -> dict:
        """
        Calculate year-over-year growth rates for each manufacturer (Vehicle Class).

        Args:
            data (pd.DataFrame): DataFrame containing 'Vehicle Class', 'Year', and 'TOTAL' columns

        Returns:
            dict: Mapping of manufacturer to YoY growth percentage
        """
        from typing import Dict
        import pandas as pd

        manufacturer_growth: Dict = {}

        try:
            # Validate required columns
            if 'Vehicle Class' not in data.columns or 'Year' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for manufacturer growth calculation")
                return manufacturer_growth

            # Ensure Year and TOTAL are numeric
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)
            data = data.dropna(subset=['Year', 'Vehicle Class'])

            # Iterate through unique manufacturers
            for manufacturer in data['Vehicle Class'].dropna().unique():
                mfg_data = data[data['Vehicle Class'] == manufacturer]

                if 'TOTAL' not in mfg_data.columns or len(mfg_data) < 2:
                    continue

                # Group by Year and sum TOTAL
                yearly_totals = mfg_data.groupby('Year', as_index=False)['TOTAL'].sum()
                yearly_totals = yearly_totals.sort_values('Year')

                # Calculate YoY growth using latest two years
                if len(yearly_totals) >= 2:
                    prev_year_val = yearly_totals.iloc[-2]['TOTAL']
                    latest_year_val = yearly_totals.iloc[-1]['TOTAL']

                    if prev_year_val > 0:
                        growth_rate = ((latest_year_val - prev_year_val) / prev_year_val) * 100
                        manufacturer_growth[manufacturer] = round(growth_rate, 2)
                    else:
                        manufacturer_growth[manufacturer] = None  # Cannot divide by zero

            self.logger.debug("✅ Manufacturer growth rates calculated")
            return manufacturer_growth

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating manufacturer growth: {e}")
            return {}
    
    def _calculate_market_share(self, data: pd.DataFrame) -> dict[str, float]:
        """
        Calculate market share (%) by manufacturer (Vehicle Class).

        Args:
            data (pd.DataFrame): DataFrame containing 'Vehicle Class' and 'TOTAL' columns

        Returns:
            dict[str, float]: Mapping of manufacturer to market share percentage
        """
        from typing import Dict
        import pandas as pd

        try:
            # Validate required columns
            if 'Vehicle Class' not in data.columns or 'TOTAL' not in data.columns:
                self.logger.warning("⚠️ Required columns missing for market share calculation")
                return {}

            # Ensure TOTAL is numeric
            data['TOTAL'] = pd.to_numeric(data['TOTAL'], errors='coerce').fillna(0)

            total_registrations = data['TOTAL'].sum()
            if total_registrations == 0:
                self.logger.warning("⚠️ Total registrations are zero. Cannot calculate market share.")
                return {}

            # Group by manufacturer and calculate market share
            manufacturer_totals = data.groupby('Vehicle Class', as_index=False)['TOTAL'].sum()
            manufacturer_totals['market_share'] = (manufacturer_totals['TOTAL'] / total_registrations * 100).round(2)

            self.logger.debug("✅ Market share calculated for all manufacturers")
            return dict(zip(manufacturer_totals['Vehicle Class'], manufacturer_totals['market_share']))

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating market share: {e}")
            return {}
    
    def get_investor_insights(self, data: pd.DataFrame = None) -> dict:
        """
        Generate key investor insights from the processed VAHAN data.

        Args:
            data (pd.DataFrame, optional): DataFrame to analyze (defaults to self.processed_data)

        Returns:
            dict: Insights including market overview, growth leaders, risk factors, and opportunities
        """
        from typing import Dict

        # Step 1: Use provided data or fallback to processed_data
        if data is None:
            data = self.processed_data

        # Step 2: Validate data availability
        if data is None or data.empty:
            self.logger.warning("⚠️ No data available to generate investor insights")
            return {}

        try:
            self.logger.info("📊 Generating investor insights...")

            insights: Dict = {
                'market_overview': self._get_market_overview(data),
                'growth_leaders': self._get_growth_leaders(),
                'risk_factors': self._identify_risk_factors(),
                'investment_opportunities': self._identify_opportunities()
            }

            self.logger.info("✅ Investor insights generated successfully")
            return insights

        except Exception as e:
            self.logger.error(f"❌ Error generating investor insights: {e}")
            return {}
    
    def _get_market_overview(self, data: pd.DataFrame) -> dict:
        """
        Generate overall market overview metrics from VAHAN data.

        Args:
            data (pd.DataFrame): DataFrame containing relevant columns ('TOTAL', 'Year', 'State', 'Vehicle_Category')

        Returns:
            dict: Market overview including total registrations, data period, states covered, and category breakdown
        """
        from typing import Dict
        import pandas as pd

        try:
            overview: Dict = {}

            # Total registrations
            if 'TOTAL' in data.columns:
                overview['total_registrations'] = int(pd.to_numeric(data['TOTAL'], errors='coerce').sum())

            # Data period (min-max year)
            if 'Year' in data.columns:
                years = pd.to_numeric(data['Year'], errors='coerce').dropna().unique()
                if len(years) > 0:
                    overview['data_period'] = f"{int(min(years))}-{int(max(years))}"

            # Number of states covered
            if 'State' in data.columns:
                overview['states_covered'] = int(data['State'].nunique())

            # Category breakdown
            if 'Vehicle_Category' in data.columns and 'TOTAL' in data.columns:
                category_totals = (
                    data.groupby('Vehicle_Category', as_index=False)['TOTAL']
                    .sum()
                )
                overview['category_breakdown'] = {
                    row['Vehicle_Category']: int(row['TOTAL']) for _, row in category_totals.iterrows()
                }

            self.logger.debug("✅ Market overview metrics generated")
            return overview

        except Exception as e:
            self.logger.warning(f"⚠️ Error getting market overview: {e}")
            return {}
    
    def _get_growth_leaders(self) -> list[str]:
        """
        Identify top growth leaders from pre-calculated growth metrics.

        Returns:
            list[str]: Descriptions of categories or states with significant growth
        """
        from typing import List

        try:
            leaders: List[str] = []

            # Check category growth
            category_growth = self.growth_metrics.get('category_growth', {})
            for category, growth_data in category_growth.items():
                if growth_data:
                    # Use latest year-over-year growth
                    latest_growth = next(reversed(growth_data.values()), 0)
                    if latest_growth is not None and latest_growth > 10:  # 10% growth threshold
                        leaders.append(f"{category} category showing {latest_growth}% growth")

            # Check state growth
            state_growth = self.growth_metrics.get('state_growth', {})
            for state, growth_data in state_growth.items():
                if growth_data:
                    latest_growth = next(reversed(growth_data.values()), 0)
                    if latest_growth is not None and latest_growth > 15:  # 15% growth threshold for states
                        leaders.append(f"{state} state showing {latest_growth}% growth")

            # Return top 5 leaders
            self.logger.debug("✅ Growth leaders identified")
            return leaders[:5]

        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying growth leaders: {e}")
            return []
    
    def _identify_risk_factors(self) -> list[str]:
        """
        Identify potential risk factors based on negative growth trends.

        Returns:
            list[str]: Descriptions of categories showing significant decline
        """
        from typing import List

        try:
            risks: List[str] = []

            category_growth = self.growth_metrics.get('category_growth', {})
            for category, growth_data in category_growth.items():
                if growth_data:
                    # Get latest YoY growth safely
                    latest_growth = next(reversed(growth_data.values()), 0)
                    if latest_growth is not None and latest_growth < -5:  # Negative growth threshold
                        risks.append(f"{category} category declining by {abs(round(latest_growth, 2))}%")

            self.logger.debug("✅ Risk factors identified")
            return risks

        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying risk factors: {e}")
            return []
    
    def _identify_opportunities(self) -> list[str]:
        """
        Identify top investment opportunities based on high manufacturer growth trends.

        Returns:
            list[str]: Descriptions of manufacturers showing exceptional growth
        """
        from typing import List

        try:
            opportunities: List[str] = []

            manufacturer_growth = self.growth_metrics.get('manufacturer_growth', {})
            growth_rates = manufacturer_growth.get('growth_rates', {})

            for manufacturer, growth_rate in growth_rates.items():
                if growth_rate is not None and growth_rate > 20:  # High growth threshold
                    opportunities.append(f"{manufacturer} showing exceptional {round(growth_rate, 2)}% growth")

            self.logger.debug("✅ Investment opportunities identified")
            return opportunities[:3]  # Top 3 opportunities

        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying opportunities: {e}")
            return []
    
    def export_processed_data(self, filename: str = None) -> str:
        """
        Export processed VAHAN data along with growth metrics to CSV files.

        Args:
            filename (str, optional): Base filename for export. Defaults to Config-generated name.

        Returns:
            str: Path to the exported main CSV file
        """
        import pandas as pd
        from pathlib import Path

        if self.processed_data is None or self.processed_data.empty:
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
                        # Flatten nested dictionaries if needed
                        if any(isinstance(v, dict) for v in metric_data.values()):
                            for key, sub_data in metric_data.items():
                                if isinstance(sub_data, dict):
                                    for sub_key, value in sub_data.items():
                                        all_metrics.append({
                                            'Metric_Type': metric_name,
                                            'Category': f"{key} - {sub_key}",
                                            'Value': value
                                        })
                        else:
                            for key, value in metric_data.items():
                                all_metrics.append({
                                    'Metric_Type': metric_name,
                                    'Category': key,
                                    'Value': value
                                })

                if all_metrics:
                    metrics_df = pd.DataFrame(all_metrics)
                    metrics_df.to_csv(metrics_filename, index=False)
                    self.logger.info(f"📊 Growth metrics exported to {metrics_filename}")

            return str(filepath)

        except Exception as e:
            raise DataProcessingError(f"Failed to export data: {e}")
    
    def process_all(self, data: pd.DataFrame) -> "ProcessingResult":
        """
        Execute the full VAHAN data processing pipeline.

        Steps:
        1. Clean the raw data
        2. Calculate growth metrics (YoY, category, state, manufacturer)
        3. Generate investor insights
        4. Return structured processing results

        Args:
            data (pd.DataFrame): Raw input VAHAN data

        Returns:
            ProcessingResult: Complete processing results including cleaned data, growth metrics,
                            insights, processing time, and record count
        """
        import time

        start_time = time.time()

        try:
            # Step 1: Clean the data
            cleaned_data = self.clean_data(data)

            # Step 2: Calculate growth metrics
            growth_metrics = self.calculate_growth_metrics(cleaned_data)

            # Step 3: Generate investor insights
            insights = self.get_investor_insights(cleaned_data)

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

            self.logger.info(f"✅ Complete processing finished in {processing_time:.2f}s "
                            f"for {len(cleaned_data)} records")
            return result

        except Exception as e:
            self.logger.error(f"❌ Complete processing failed: {e}")
            raise DataProcessingError(f"Complete processing failed: {e}")
