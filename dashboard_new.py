"""
Streamlit Dashboard for VAHAN Vehicle Registration Data Analysis
Updated to use the new modular architecture.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.scrapers import VahanScraper
from src.processors import VahanDataProcessor
from src.analytics import GrowthAnalyzer, InsightGenerator
from src.utils import setup_logging, get_logger, create_sample_data
from src.core.models import FilterCombination

import streamlit as st
from src.utils.logging_utils import setup_logging
from datetime import datetime

# ──────────────────────────────
# Page configuration
# ──────────────────────────────
st.set_page_config(
    page_title="VAHAN Vehicle Registration Analytics 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────
# Setup logging
# ──────────────────────────────
setup_logging()

# ──────────────────────────────
# Custom CSS for investor-focused styling
# ──────────────────────────────
st.markdown("""
<style>
    /* Main page header */
    .main-header {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #1f77b4 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6 !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border-left: 6px solid #1f77b4 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* Positive growth metrics */
    .growth-positive {
        color: #28a745 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
    }

    /* Negative growth metrics */
    .growth-negative {
        color: #dc3545 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
    }

    /* Insight box */
    .insight-box {
        background-color: #e8f4f8 !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin: 1rem 0 !important;
        border-left: 5px solid #17a2b8 !important;
        font-size: 1rem !important;
    }

    /* Sidebar header */
    .sidebar .sidebar-content h2 {
        font-size: 1.5rem !important;
        color: #1f77b4 !important;
        font-weight: bold !important;
        margin-bottom: 1rem !important;
    }

    /* Streamlit buttons */
    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #145a86 !important;
    }

    /* Footer timestamp */
    .footer-timestamp {
        font-size: 0.85rem !important;
        color: #666 !important;
        text-align: right !important;
        margin-top: 2rem !important;
        font-style: italic !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────
# Page Header
# ──────────────────────────────
st.markdown('<div class="main-header">VAHAN Vehicle Registration Analytics</div>', unsafe_allow_html=True)

# Optional: footer timestamp for data recency
st.markdown(f'<div class="footer-timestamp">Page generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)

class ModularVahanDashboard:
    """Modular VAHAN Dashboard using the new architecture."""
    
    def __init__(self):
        """Initialize the VAHAN dashboard with modular components and session state."""
        # ──────────────────────────────
        # Core processing modules
        # ──────────────────────────────
        self.processor = VahanDataProcessor()
        self.growth_analyzer = GrowthAnalyzer()
        self.insight_generator = InsightGenerator()
    
        # ──────────────────────────────
        # Logger for debugging and analytics
        # ──────────────────────────────
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("✅ Dashboard initialized with processor, growth analyzer, and insight generator")
    
        # ──────────────────────────────
        # Session state initialization for Streamlit
        # ──────────────────────────────
        session_defaults = {
            'scraped_data': None,
            'scraped_growth_metrics': {},
            'data_source_type': None,
            'cached_dropdown_data': None,
            'dropdown_cache_timestamp': None,
            'selected_filters': {},        # Track current filter selection
            'last_scrape_timestamp': None, # Track last data fetch
            'insights_generated': False    # Track whether insights were computed
        }
    
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                self.logger.debug(f"Initialized session state: {key} -> {default_value}")
    
        # ──────────────────────────────
        # Cache management for dropdowns
        # ──────────────────────────────
        self.logger.info("ℹ️ Dashboard session state setup complete, ready for interaction")

    def load_data(self, data_source="sample") -> bool:
        """
        Load VAHAN data via sample generation, file upload, or live scraping.
    
        Args:
            data_source (str): 'sample', 'upload', or 'scrape'.
    
        Returns:
            bool: True if data is loaded and processed successfully, else False.
        """
        # ──────────────────────────────
        # Use cached scraped data if available
        # ──────────────────────────────
        if (
            data_source == "scrape"
            and st.session_state.scraped_data is not None
            and st.session_state.data_source_type == "scrape"
        ):
            self.data = st.session_state.scraped_data
            self.growth_metrics = st.session_state.scraped_growth_metrics
            st.success("✅ Using previously scraped data from session cache!")
            self.logger.info("Loaded cached scraped data")
            return True
    
        # ──────────────────────────────
        # Load sample data
        # ──────────────────────────────
        if data_source == "sample":
            self.data = create_sample_data()
            st.session_state.data_source_type = "sample"
            st.success("✅ Sample VAHAN data loaded successfully!")
            self.logger.info("Sample data loaded")
        
        # ──────────────────────────────
        # Load user-uploaded CSV
        # ──────────────────────────────
        elif data_source == "upload":
            uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                try:
                    self.data = pd.read_csv(uploaded_file)
                    st.session_state.data_source_type = "upload"
                    st.success(f"✅ Uploaded file loaded successfully! Rows: {len(self.data)}")
                    self.logger.info(f"Uploaded file loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Failed to load uploaded file: {e}")
                    self.logger.error(f"Error loading uploaded CSV: {e}")
                    return False
            else:
                st.info("👆 Please upload a CSV file to continue.")
                return False
        
        # ──────────────────────────────
        # Scrape live data
        # ──────────────────────────────
        elif data_source == "scrape":
            return self.scrape_live_data()
    
        # ──────────────────────────────
        # Process and cache the data
        # ──────────────────────────────
        if hasattr(self, "data") and self.data is not None:
            try:
                result = self.processor.process_all(self.data)
                self.data = result.cleaned_data
                self.growth_metrics = result.growth_metrics.__dict__ if result.growth_metrics else {}
    
                # Cache scraped data
                if data_source == "scrape":
                    st.session_state.scraped_data = self.data
                    st.session_state.scraped_growth_metrics = self.growth_metrics
                    st.session_state.last_scrape_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
                st.success(f"✅ Data processed successfully! Rows: {len(self.data)}, Columns: {len(self.data.columns)}")
                self.logger.info("Data processed successfully")
                return True
            except Exception as e:
                st.error(f"❌ Error processing data: {e}")
                self.logger.error(f"Data processing failed: {e}")
                return False
    
        # ──────────────────────────────
        # If all else fails
        # ──────────────────────────────
        st.warning("⚠️ No data loaded")
        self.logger.warning("No data available after load attempt")
        return False
    
    def get_cached_dropdown_data(self, max_retries: int = 3) -> dict:
        """
        Get cached dropdown data or scrape fresh if cache is missing or stale.
    
        Args:
            max_retries (int): Number of retry attempts for scraping dropdowns.
    
        Returns:
            dict: Dropdown data mapping or empty dict if failed.
        """
        from datetime import datetime, timedelta
    
        # ──────────────────────────────
        # Use valid cached data if less than 1 hour old
        # ──────────────────────────────
        cached_data = st.session_state.get("cached_dropdown_data")
        cache_timestamp = st.session_state.get("dropdown_cache_timestamp")
    
        if cached_data and cache_timestamp:
            if datetime.now() - cache_timestamp < timedelta(hours=1):
                st.info("📋 Using cached dropdown data (refreshed within last hour)")
                self.logger.info("Returning cached dropdown data")
                return cached_data
    
        # ──────────────────────────────
        # Scrape fresh dropdown data
        # ──────────────────────────────
        st.info("🔄 Fetching fresh dropdown data from VAHAN website...")
        scraper = VahanScraper()
        dropdown_data = {}
    
        for attempt in range(1, max_retries + 1):
            try:
                with st.spinner(f"🔄 Attempt {attempt}: Initializing browser..."):
                    scraper.setup_driver(headless=True)
                    scraper.open_page()
    
                with st.spinner(f"📋 Attempt {attempt}: Scraping dropdown options..."):
                    dropdown_data = scraper.scrape_dropdowns()
    
                if dropdown_data:
                    st.session_state.cached_dropdown_data = dropdown_data
                    st.session_state.dropdown_cache_timestamp = datetime.now()
                    st.success("✅ Dropdown data fetched and cached successfully!")
                    self.logger.info("Dropdown data cached successfully")
                    return dropdown_data
                else:
                    st.warning(f"⚠️ Attempt {attempt}: No dropdown data found")
                    self.logger.warning(f"Attempt {attempt}: Empty dropdown data")
    
            except Exception as e:
                st.error(f"❌ Attempt {attempt}: Error fetching dropdowns: {e}")
                self.logger.error(f"Attempt {attempt}: Dropdown scraping failed: {e}")
    
            finally:
                try:
                    scraper.close()
                except:
                    pass
    
        st.error("❌ Failed to fetch dropdown data after multiple attempts")
        self.logger.error("All dropdown scraping attempts failed")
        return {}
    
    def scrape_live_data(self, max_retries: int = 3) -> bool:
        """Scrape live VAHAN data with caching, progress, and processing."""
        st.info("🌐 Preparing to scrape live data from VAHAN website...")
    
        # Refresh cache button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("📋 Using cached dropdown data for performance")
        with col2:
            if st.button("🔄 Refresh Cache"):
                st.session_state.cached_dropdown_data = None
                st.session_state.dropdown_cache_timestamp = None
                st.rerun()
    
        # Get dropdown data (cached or fresh)
        dropdown_data = self.get_cached_dropdown_data()
        if not dropdown_data:
            st.error("❌ Could not fetch dropdown options.")
            return False
    
        st.success("✅ Dropdown options loaded successfully!")
    
        # ──────────────
        # User selection
        # ──────────────
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📍 Select States")
                available_states = dropdown_data.get("State", [])
                selected_states = st.multiselect("States:", available_states, default=[])
    
            with col2:
                st.subheader("📅 Select Years")
                available_years = dropdown_data.get("Year", [])
                selected_years = st.multiselect("Years:", available_years, default=[])
    
            st.subheader("🚗 Select Vehicle Types")
            available_vehicle_types = dropdown_data.get("Vehicle Type", [])
            selected_vehicle_types = st.multiselect(
                "Vehicle Types:", available_vehicle_types, default=[]
            )
    
            total_combinations = (
                len(selected_states) * len(selected_years) *
                (len(selected_vehicle_types) if selected_vehicle_types else 1)
            )
    
            st.info(f"Total combinations to scrape: {total_combinations}")
    
            if total_combinations == 0:
                st.warning("❌ Select at least one state and one year to continue.")
                return False
    
        except Exception as selection_error:
            st.error(f"❌ Failed to read user selections: {selection_error}")
            return False
    
        # ──────────────
        # Start scraping
        # ──────────────
        if st.button("🚀 Start Scraping", type="primary"):
            combinations = []
            for state in selected_states:
                for year in selected_years:
                    if selected_vehicle_types:
                        for vehicle in selected_vehicle_types:
                            combinations.append({
                                "State": state,
                                "Year": year,
                                "Vehicle Type": vehicle
                            })
                    else:
                        combinations.append({"State": state, "Year": year})
    
            progress_bar = st.progress(0)
            status_text = st.empty()
    
            for attempt in range(1, max_retries + 1):
                try:
                    scraper = VahanScraper()
                    scraper.setup_driver(headless=True)
                    scraper.open_page()
    
                    status_text.text(f"🔄 Scraping attempt {attempt}/{max_retries}...")
                    scraped_data = scraper.scrape_multiple_combinations(combinations)
    
                    if scraped_data.empty:
                        st.warning(f"⚠️ Attempt {attempt}: No data scraped.")
                        continue
    
                    progress_bar.progress(80)
                    status_text.text("🔄 Processing scraped data...")
    
                    processed_data = self.processor.clean_data(scraped_data)
                    self.growth_metrics = self.processor.calculate_growth_metrics(processed_data)
    
                    self.data = processed_data
                    st.session_state.scraped_data = processed_data.copy()
                    st.session_state.scraped_growth_metrics = self.growth_metrics.copy()
                    st.session_state.data_source_type = "scrape"
    
                    progress_bar.progress(100)
                    status_text.text("✅ Scraping and processing completed successfully!")
                    st.success(f"🎉 Scraped and processed {len(processed_data)} records!")
    
                    with st.expander("📋 Preview Processed Data"):
                        st.dataframe(processed_data.head(10))
                        if "TOTAL" in processed_data.columns:
                            st.info(f"📊 Total registrations: {processed_data['TOTAL'].sum():,}")
    
                    return True
    
                except Exception as e:
                    st.error(f"❌ Attempt {attempt} failed: {e}")
                finally:
                    try:
                        scraper.close()
                    except:
                        pass
    
            st.error("❌ All scraping attempts failed. Please try again later.")
            return False
    
    def create_sidebar_filters(self) -> dict:
        """Create sidebar filters for processed VAHAN data."""
        st.sidebar.header("🔍 Data Filters")
        filters = {}
    
        # ──────────────
        # Data Source Indicator
        # ──────────────
        source_type = st.session_state.get("data_source_type")
        if source_type:
            if source_type == "scrape":
                st.sidebar.success("🌐 Live Scraped Data")
                if st.sidebar.button("🗑️ Clear Scraped Data"):
                    st.session_state.scraped_data = None
                    st.session_state.scraped_growth_metrics = {}
                    st.session_state.data_source_type = None
                    st.rerun()
            elif source_type == "sample":
                st.sidebar.info("🧪 Sample Data")
            elif source_type == "upload":
                st.sidebar.info("📁 Uploaded Data")
    
        # ──────────────
        # Year Filter
        # ──────────────
        if "Year" in self.data.columns:
            years = sorted(self.data["Year"].dropna().unique())
            if years:
                filters["years"] = st.sidebar.multiselect(
                    "📅 Select Years",
                    options=years,
                    default=years,
                    help="Filter data by year(s)"
                )
    
        # ──────────────
        # State Filter
        # ──────────────
        if "State" in self.data.columns:
            states = sorted(self.data["State"].dropna().unique())
            if states:
                default_states = states[:10] if len(states) > 10 else states
                filters["states"] = st.sidebar.multiselect(
                    "📍 Select States",
                    options=states,
                    default=default_states,
                    help="Filter data by state(s). Showing top 10 by default for performance."
                )
    
        # ──────────────
        # Vehicle Category Filter
        # ──────────────
        if "Vehicle_Category" in self.data.columns:
            categories = sorted(self.data["Vehicle_Category"].dropna().unique())
            if categories:
                filters["categories"] = st.sidebar.multiselect(
                    "🚗 Select Vehicle Categories",
                    options=categories,
                    default=categories,
                    help="Filter data by vehicle category"
                )
    
        # ──────────────
        # Optional: Display selected filters for clarity
        # ──────────────
        if filters:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔹 Selected Filters Summary")
            for key, val in filters.items():
                st.sidebar.text(f"{key.capitalize()}: {len(val)} selected")
    
        return filters
    
    def apply_filters(self, filters: dict) -> pd.DataFrame:
        """
        Apply selected filters to the dataset.
    
        Args:
            filters (dict): Dictionary containing filter selections:
                - 'years': List of years
                - 'states': List of states
                - 'categories': List of vehicle categories
    
        Returns:
            pd.DataFrame: Filtered dataset
        """
        filtered_data = self.data.copy()
        
        # Year filter
        if 'years' in filters and filters['years']:
            if 'Year' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Year'].isin(filters['years'])]
            else:
                self.logger.warning("⚠️ 'Year' column not found in data, skipping year filter")
        
        # State filter
        if 'states' in filters and filters['states']:
            if 'State' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['State'].isin(filters['states'])]
            else:
                self.logger.warning("⚠️ 'State' column not found in data, skipping state filter")
        
        # Vehicle category filter
        if 'categories' in filters and filters['categories']:
            if 'Vehicle_Category' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Vehicle_Category'].isin(filters['categories'])]
            else:
                self.logger.warning("⚠️ 'Vehicle_Category' column not found in data, skipping category filter")
        
        self.logger.info(f"✅ Filters applied: {filters}. Resulting rows: {len(filtered_data)}")
        return filtered_data
    
    def create_kpi_cards(self, data: pd.DataFrame):
        """Display fully maxed investor-focused KPI dashboard with analytics and trends."""
        st.subheader("📊 Key Performance Indicators & Insights")
    
        # ----------------------------
        # Prepare Data
        # ----------------------------
        total_registrations = 0
        states_count = 0
        categories_count = 0
        growth_text = "N/A"
        growth_color = ""
        period_text = "Unknown"
        help_text = "Data period information not available"
        
        # Total registrations
        if 'TOTAL' in data.columns:
            try:
                total_registrations = int(data['TOTAL'].sum())
                formatted_total = f"{total_registrations:,}"
            except Exception:
                formatted_total = "Processing..."
        else:
            formatted_total = "0"
    
        # States count
        for col_name in ['State', 'Filter_State']:
            if col_name in data.columns:
                states_count = data[col_name].nunique()
                break
    
        # Categories count
        for col_name in ['Vehicle_Category', 'Vehicle Class']:
            if col_name in data.columns:
                categories_count = data[col_name].nunique()
                break
    
        # Data period
        year_col = None
        for col_name in ['Year', 'Filter_Year']:
            if col_name in data.columns:
                year_col = col_name
                break
        if year_col:
            years_span = data[year_col].nunique()
            min_year, max_year = data[year_col].min(), data[year_col].max()
            period_text = f"{years_span} years ({min_year}-{max_year})" if years_span > 1 else f"{min_year}"
            help_text = f"Data spans from {min_year} to {max_year}"
    
        # Average YoY growth
        growth_rate = None
        if year_col and 'TOTAL' in data.columns and data[year_col].nunique() > 1:
            try:
                yearly_totals = data.groupby(year_col)['TOTAL'].sum().sort_index()
                latest_year, previous_year = yearly_totals.index[-1], yearly_totals.index[-2]
                growth_rate = ((yearly_totals[latest_year] - yearly_totals[previous_year])
                               / yearly_totals[previous_year]) * 100
                growth_text = f"{growth_rate:.1f}%"
                growth_color = "growth-positive" if growth_rate >= 0 else "growth-negative"
            except Exception:
                growth_text = "N/A"
    
        # ----------------------------
        # KPI Cards
        # ----------------------------
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            st.metric("Total Registrations", formatted_total, help="Total vehicle registrations in the dataset")
            if 'Year' in data.columns and 'TOTAL' in data.columns:
                st.line_chart(data.groupby('Year')['TOTAL'].sum().sort_index(), height=100)
    
        with col2:
            st.metric("States Covered", states_count, help="Number of unique states in the dataset")
            if 'State' in data.columns:
                top_states = data['State'].value_counts().head(5)
                st.bar_chart(top_states, height=100)
    
        with col3:
            st.markdown(f'<div class="{growth_color}"><strong>Avg YoY Growth:</strong> {growth_text}</div>', unsafe_allow_html=True)
            if growth_rate is not None:
                st.line_chart(data.groupby(year_col)['TOTAL'].sum().pct_change().fillna(0) * 100, height=100)
    
        with col4:
            st.metric("Data Period", period_text, help=help_text)
            if 'Vehicle_Category' in data.columns:
                top_categories = data['Vehicle_Category'].value_counts().head(5)
                st.bar_chart(top_categories, height=100)
            elif 'Vehicle Class' in data.columns:
                top_categories = data['Vehicle Class'].value_counts().head(5)
                st.bar_chart(top_categories, height=100)
    
        # ----------------------------
        # Custom CSS for growth coloring
        # ----------------------------
        st.markdown("""
        <style>
            .growth-positive { color: #28a745; font-weight: bold; font-size:1.1rem; }
            .growth-negative { color: #dc3545; font-weight: bold; font-size:1.1rem; }
        </style>
        """, unsafe_allow_html=True)
    
        # ----------------------------
        # Optional summary stats
        # ----------------------------
        with st.expander("📋 Summary Details"):
            st.write(data.describe(include='all').T)
            if 'TOTAL' in data.columns:
                st.info(f"Total vehicle registrations in dataset: {total_registrations:,}")
            if 'State' in data.columns:
                st.info(f"Number of unique states: {states_count}")
            if 'Vehicle_Category' in data.columns or 'Vehicle Class' in data.columns:
                st.info(f"Number of vehicle categories: {categories_count}")
    
    def create_growth_charts(self, data: pd.DataFrame):
        """📈 Fully maxed Growth Charts and Investor Insights."""
        st.header("📈 Growth Analysis & Trends")
    
        # ----------------------------
        # Yearly Trend and CAGR
        # ----------------------------
        if 'Year' in data.columns and 'TOTAL' in data.columns and not data.empty:
            try:
                # CAGR & Volatility
                cagr = self.growth_analyzer.calculate_compound_growth_rate(data)
                volatility = self.growth_analyzer.analyze_growth_volatility(data)
                
                col1, col2 = st.columns([2, 1])
    
                with col1:
                    yearly_data = data.groupby('Year')['TOTAL'].sum().reset_index()
                    fig = px.line(
                        yearly_data, x='Year', y='TOTAL', markers=True,
                        title="Total Registrations Trend"
                    )
                    fig.update_layout(
                        height=400,
                        xaxis_title="Year",
                        yaxis_title="Total Registrations",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
                with col2:
                    st.metric("CAGR", f"{cagr:.2f}%", help="Compound Annual Growth Rate")
                    if volatility and 'stability_score' in volatility:
                        st.metric("Stability", volatility['stability_score'], help="Stability of growth over years")
    
            except Exception as e:
                st.warning(f"⚠️ Error creating yearly growth chart: {e}")
                st.info(f"📊 Available columns: {', '.join(data.columns.tolist())}")
    
        # ----------------------------
        # Vehicle Category-wise Growth
        # ----------------------------
        st.subheader("📊 Growth by Vehicle Category")
        
        vehicle_col = None
        year_col = None
        for col in ['Vehicle_Category', 'Vehicle Class', 'Vehicle_Type']:
            if col in data.columns:
                vehicle_col = col
                break
        for col in ['Year', 'Filter_Year']:
            if col in data.columns:
                year_col = col
                break
    
        if vehicle_col and year_col and data[year_col].nunique() > 1:
            try:
                cat_yearly = data.groupby([year_col, vehicle_col])['TOTAL'].sum().reset_index()
                if not cat_yearly.empty:
                    fig = px.line(
                        cat_yearly, x=year_col, y='TOTAL', color=vehicle_col, markers=True,
                        title="Growth by Vehicle Category"
                    )
                    fig.update_layout(
                        height=400,
                        xaxis_title="Year",
                        yaxis_title="Total Registrations",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 No data available for category-wise growth analysis")
            except Exception as e:
                st.warning(f"⚠️ Error creating category growth chart: {e}")
                st.info("📊 Available columns: " + ", ".join(data.columns.tolist()))
    
        elif vehicle_col and year_col:
            # Single year distribution
            st.info("📊 Single year data - current vehicle category distribution")
            category_totals = data.groupby(vehicle_col)['TOTAL'].sum().reset_index()
            category_totals = category_totals.sort_values('TOTAL', ascending=False)
            fig = px.bar(
                category_totals, x=vehicle_col, y='TOTAL', text='TOTAL',
                title=f"{vehicle_col} Distribution (Current Year)"
            )
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400, xaxis_tickangle=-45, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Cannot create growth chart: Missing vehicle category or year data")
            st.info(f"📊 Available columns: {', '.join(data.columns.tolist())}")
            if vehicle_col:
                st.info(f"✅ Vehicle column found: {vehicle_col}")
            if year_col:
                st.info(f"✅ Year column found: {year_col}")
            st.info(f"📅 Unique years: {data[year_col].nunique() if year_col else 'N/A'}")
    
    def create_market_share_analysis(self, data: pd.DataFrame):
        """🥧 Market Composition & Share Analysis - Fully Maxed Investor View"""
        st.subheader("🥧 Market Composition & Share Analysis")
    
        col1, col2 = st.columns(2)
    
        # ----------------------------
        # 1️⃣ Market Share by Vehicle Category/Class
        # ----------------------------
        with col1:
            vehicle_col = None
            for col in ['Vehicle Class', 'Vehicle_Category', 'Vehicle_Type']:
                if col in data.columns:
                    vehicle_col = col
                    break
    
            if vehicle_col:
                vehicle_totals = data.groupby(vehicle_col)['TOTAL'].sum().reset_index()
                vehicle_totals = vehicle_totals.sort_values('TOTAL', ascending=False)
    
                fig_pie = px.pie(
                    vehicle_totals,
                    values='TOTAL',
                    names=vehicle_col,
                    title=f'Market Share by {vehicle_col}',
                    hole=0.3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(vehicle_totals))
                fig_pie.update_layout(
                    height=450,
                    legend_title=f"{vehicle_col}",
                    title_font_size=18
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("⚠️ No vehicle category/class column found for market share analysis.")
    
        # ----------------------------
        # 2️⃣ Top States by Registrations
        # ----------------------------
        with col2:
            state_col = None
            for col in ['State', 'Filter_State']:
                if col in data.columns:
                    state_col = col
                    break
    
            if state_col:
                state_totals = data.groupby(state_col)['TOTAL'].sum().reset_index()
                state_totals = state_totals.sort_values('TOTAL', ascending=False).head(10)
    
                fig_states = px.bar(
                    state_totals,
                    x='TOTAL',
                    y=state_col,
                    orientation='h',
                    text='TOTAL',
                    title='Top 10 States by Registrations',
                    color='TOTAL',
                    color_continuous_scale='Viridis'
                )
                fig_states.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig_states.update_layout(
                    height=500,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Total Registrations",
                    yaxis_title=state_col,
                    title_font_size=18
                )
                st.plotly_chart(fig_states, use_container_width=True)
            else:
                st.info("⚠️ No state column found for top state analysis.")
    
        # ----------------------------
        # Optional Additional Insights
        # ----------------------------
        if vehicle_col and state_col:
            st.markdown(
                f"ℹ️ Market share breakdown shows distribution of {vehicle_col} across the top states by registrations."
            )
    
    def create_time_series_analysis(self, data: pd.DataFrame):
        """📅 Time Series Analysis - Fully Maxed Investor View"""
        st.subheader("📅 Time Series Analysis")
    
        # Determine vehicle and year columns dynamically
        vehicle_col = None
        for col in ['Vehicle Class', 'Vehicle_Category', 'Vehicle_Type']:
            if col in data.columns:
                vehicle_col = col
                break
    
        year_col = None
        for col in ['Year', 'Filter_Year']:
            if col in data.columns:
                year_col = col
                break
    
        if not vehicle_col or not year_col:
            st.warning("⚠️ Cannot create time series analysis. Missing vehicle or year data.")
            st.info(f"Available columns: {', '.join(data.columns.tolist())}")
            return
    
        # Check if multiple years are available
        unique_years = data[year_col].nunique()
        if unique_years > 1:
            # ----------------------------
            # Line Chart: Yearly Trends by Vehicle
            # ----------------------------
            yearly_vehicle = data.groupby([year_col, vehicle_col])['TOTAL'].sum().reset_index()
    
            fig_timeline = px.line(
                yearly_vehicle,
                x=year_col,
                y='TOTAL',
                color=vehicle_col,
                title=f'Registration Trends by {vehicle_col} Over Time',
                markers=True
            )
            fig_timeline.update_layout(
                height=500,
                xaxis_title="Year",
                yaxis_title="Total Registrations",
                legend_title=vehicle_col,
                title_font_size=18
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
            # ----------------------------
            # Heatmap: Registration Intensity
            # ----------------------------
            heatmap_data = yearly_vehicle.pivot(index=vehicle_col, columns=year_col, values='TOTAL')
            fig_heatmap = px.imshow(
                heatmap_data,
                title=f'Registration Intensity Heatmap ({vehicle_col} vs {year_col})',
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
        else:
            # ----------------------------
            # Single Year Distribution
            # ----------------------------
            st.info(f"📊 Single year data - showing {vehicle_col} distribution")
            vehicle_data = data.groupby(vehicle_col)['TOTAL'].sum().reset_index()
            vehicle_data = vehicle_data.sort_values('TOTAL', ascending=False)
    
            fig_single = px.bar(
                vehicle_data,
                x=vehicle_col,
                y='TOTAL',
                title=f'{vehicle_col} Distribution (Current Year)',
                text='TOTAL'
            )
            fig_single.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_single.update_layout(
                height=400,
                xaxis_tickangle=-45,
                margin=dict(b=100),
                yaxis_title="Total Registrations",
                xaxis_title=vehicle_col,
                title_font_size=18
            )
            st.plotly_chart(fig_single, use_container_width=True)
    
        # Optional insights
        st.markdown(
            f"ℹ️ Time series analysis shows trends and intensity of registrations across {vehicle_col} over {year_col}."
        )
    
    def create_manufacturer_analysis(self, data: pd.DataFrame):
        """🏭 Manufacturer & Vehicle Class Analysis - Investor Maxed"""
        st.subheader("🏭 Manufacturer & Vehicle Class Analysis")
    
        vehicle_col = None
        for col in ['Vehicle Class', 'Vehicle_Category', 'Vehicle_Type']:
            if col in data.columns:
                vehicle_col = col
                break
    
        if not vehicle_col:
            st.warning("⚠️ Cannot create manufacturer analysis. No vehicle class column found.")
            st.info(f"Available columns: {', '.join(data.columns.tolist())}")
            return
    
        # Top vehicle classes / manufacturers
        manufacturer_totals = data.groupby(vehicle_col)['TOTAL'].sum().reset_index()
        manufacturer_totals = manufacturer_totals.sort_values('TOTAL', ascending=False).head(15)
    
        col1, col2 = st.columns(2)
    
        # ----------------------------
        # Left: Horizontal Bar Chart
        # ----------------------------
        with col1:
            fig_manufacturers = px.bar(
                manufacturer_totals,
                x='TOTAL',
                y=vehicle_col,
                orientation='h',
                title=f'Top 15 {vehicle_col} by Registrations',
                text='TOTAL',
                color='TOTAL',
                color_continuous_scale='Blues'
            )
            fig_manufacturers.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_manufacturers.update_layout(
                height=600,
                xaxis_title="Total Registrations",
                yaxis_title=vehicle_col,
                title_font_size=18,
                margin=dict(l=0, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_manufacturers, use_container_width=True)
    
        # ----------------------------
        # Right: Data Table
        # ----------------------------
        with col2:
            st.subheader(f"📊 Top {vehicle_col} Table")
            display_data = manufacturer_totals.copy()
            display_data['TOTAL'] = display_data['TOTAL'].apply(lambda x: f"{x:,}")
            st.dataframe(display_data, use_container_width=True)
    
        st.markdown(
            f"ℹ️ This analysis highlights the top {vehicle_col} in terms of registrations, "
            "giving investors a clear view of market concentration and leading vehicle types."
        )
    
    def create_comparison_tool(self, data: pd.DataFrame):
        """🔄 Interactive Comparison Tool - Maxed for Investors"""
        st.subheader("🔄 Interactive Comparison Tool")
    
        col1, col2 = st.columns(2)
    
        # ----------------------------
        # Left: Compare States
        # ----------------------------
        with col1:
            st.markdown("**📍 Compare States**")
    
            state_col = None
            for col_name in ['State', 'Filter_State']:
                if col_name in data.columns:
                    state_col = col_name
                    break
    
            if not state_col:
                st.warning("⚠️ No state column available for comparison")
            else:
                states_to_compare = st.multiselect(
                    "Select states to compare",
                    options=sorted(data[state_col].unique()),
                    default=sorted(data[state_col].unique())[:3]
                )
    
                if states_to_compare:
                    comparison_data = data[data[state_col].isin(states_to_compare)]
    
                    if 'Year' in data.columns:
                        # Line chart for state-wise trends
                        yearly_comparison = comparison_data.groupby(['Year', state_col])['TOTAL'].sum().reset_index()
                        fig_compare = px.line(
                            yearly_comparison,
                            x='Year',
                            y='TOTAL',
                            color=state_col,
                            markers=True,
                            title="📈 State-wise Registration Trends Over Time"
                        )
                        fig_compare.update_layout(
                            height=400,
                            xaxis_title="Year",
                            yaxis_title="Total Registrations",
                            legend_title="State"
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    else:
                        # Bar chart fallback if no Year column
                        state_totals = comparison_data.groupby(state_col)['TOTAL'].sum().reset_index()
                        state_totals = state_totals.sort_values('TOTAL', ascending=False)
                        fig_compare = px.bar(
                            state_totals,
                            x=state_col,
                            y='TOTAL',
                            title="📊 State-wise Registration Comparison",
                            text='TOTAL'
                        )
                        fig_compare.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                        fig_compare.update_layout(height=400)
                        st.plotly_chart(fig_compare, use_container_width=True)
    
        # ----------------------------
        # Right: Compare Vehicle Classes
        # ----------------------------
        with col2:
            st.markdown("**🚗 Compare Vehicle Classes**")
    
            vehicle_col = None
            for col_name in ['Vehicle Class', 'Vehicle_Category', 'Vehicle_Type']:
                if col_name in data.columns:
                    vehicle_col = col_name
                    break
    
            if not vehicle_col:
                st.warning("⚠️ No vehicle class column available for comparison")
            else:
                vehicle_classes_to_compare = st.multiselect(
                    "Select vehicle classes to compare",
                    options=sorted(data[vehicle_col].unique()),
                    default=sorted(data[vehicle_col].unique())[:3] if len(data[vehicle_col].unique()) >= 3 else sorted(data[vehicle_col].unique())
                )
    
                if vehicle_classes_to_compare:
                    vehicle_comparison = data[data[vehicle_col].isin(vehicle_classes_to_compare)]
                    vehicle_totals = vehicle_comparison.groupby(vehicle_col)['TOTAL'].sum().reset_index()
                    vehicle_totals = vehicle_totals.sort_values('TOTAL', ascending=False)
    
                    fig_vehicle_bar = px.bar(
                        vehicle_totals,
                        x=vehicle_col,
                        y='TOTAL',
                        title=f"📊 {vehicle_col} Comparison",
                        text='TOTAL',
                        color='TOTAL',
                        color_continuous_scale='Blues'
                    )
                    fig_vehicle_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_vehicle_bar.update_layout(
                        height=400,
                        xaxis_title=vehicle_col,
                        yaxis_title="Total Registrations",
                        xaxis_tickangle=-45,
                        margin=dict(b=100)
                    )
                    st.plotly_chart(fig_vehicle_bar, use_container_width=True)
    
        st.markdown(
            "ℹ️ Use this interactive comparison tool to visualize and analyze "
            "state-wise and vehicle class registration trends, providing key insights for investment decisions."
        )
    
    def create_investor_insights(self, data: pd.DataFrame):
        """💡 Generate and display investor-focused insights with maxed formatting."""
        st.subheader("💡 Investor Insights & Analysis")
        
        col1, col2 = st.columns(2)
    
        # ----------------------------
        # Left Column: Market Overview
        # ----------------------------
        with col1:
            st.markdown("### 📊 Market Overview")
    
            total_registrations = data['TOTAL'].sum() if 'TOTAL' in data.columns else 0
    
            # Top Vehicle Class
            vehicle_col = None
            for col_name in ['Vehicle Class', 'Vehicle_Category', 'Vehicle_Type']:
                if col_name in data.columns:
                    vehicle_col = col_name
                    break
            if vehicle_col:
                top_vehicle = data.groupby(vehicle_col)['TOTAL'].sum().idxmax()
                st.info(f"🚗 **Top Vehicle Class**: {top_vehicle}")
    
            # Top State
            state_col = None
            for col_name in ['State', 'Filter_State']:
                if col_name in data.columns:
                    state_col = col_name
                    break
            if state_col:
                top_state = data.groupby(state_col)['TOTAL'].sum().idxmax()
                st.info(f"🏆 **Top State**: {top_state}")
    
            # Formatted total registrations
            try:
                total_registrations = int(float(total_registrations))
                formatted_total = f"{total_registrations:,}"
            except (ValueError, TypeError):
                formatted_total = str(total_registrations)
    
            st.metric(
                "💰 Total Market Size",
                formatted_total,
                help="Total number of registered vehicles in the dataset"
            )
    
        # ----------------------------
        # Right Column: Key Insights
        # ----------------------------
        with col2:
            st.markdown("### 🎯 Key Insights")
    
            # Year-over-year growth
            if 'Year' in data.columns and data['Year'].nunique() > 1:
                yearly_totals = data.groupby('Year')['TOTAL'].sum().sort_index()
                if len(yearly_totals) >= 2:
                    latest_growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2]) * 100
                    if latest_growth > 0:
                        st.success(f"📈 **Growth**: +{latest_growth:.1f}% YoY")
                    else:
                        st.warning(f"📉 **Decline**: {latest_growth:.1f}% YoY")
    
            # Market concentration (Top 3 vehicle classes)
            if vehicle_col:
                vehicle_totals = data.groupby(vehicle_col)['TOTAL'].sum().sort_values(ascending=False)
                if vehicle_totals.sum() > 0:
                    top_3_share = vehicle_totals.head(3).sum() / vehicle_totals.sum() * 100
                    st.info(f"🎯 **Top 3 Classes Market Share**: {top_3_share:.1f}%")
    
            # Optional: highlight highest growth vehicle class
            if vehicle_col and 'Year' in data.columns:
                growth_per_class = data.groupby(['Year', vehicle_col])['TOTAL'].sum().unstack(fill_value=0)
                if growth_per_class.shape[0] >= 2:
                    latest_year, prev_year = growth_per_class.index[-1], growth_per_class.index[-2]
                    growth_rate_class = ((growth_per_class.loc[latest_year] - growth_per_class.loc[prev_year]) /
                                         growth_per_class.loc[prev_year].replace(0, 1)) * 100
                    fastest_growing = growth_rate_class.idxmax()
                    st.info(f"🚀 **Fastest Growing Vehicle Class**: {fastest_growing} ({growth_rate_class.max():.1f}% YoY)")
    
    def create_export_section(self, data: pd.DataFrame):
        """📥 Create data export and download functionality with maxed features."""
        st.subheader("📥 Export & Download")
    
        col1, col2, col3 = st.columns(3)
    
        # ----------------------------
        # 1️⃣ Export Processed Data
        # ----------------------------
        with col1:
            if st.button("📊 Export Processed Data"):
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="💾 Download Processed Data CSV",
                    data=csv_data,
                    file_name=f"vahan_processed_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.success("✅ Processed data ready for download!")
    
        # ----------------------------
        # 2️⃣ Export Summary Report
        # ----------------------------
        with col2:
            if st.button("📈 Export Summary Report"):
                # Create summary report
                summary_text = f"""
    VAHAN Vehicle Registration Analysis Report
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    📊 SUMMARY STATISTICS:
    - Total Records: {len(data):,}
    - Total Registrations: {data['TOTAL'].sum() if 'TOTAL' in data.columns else 0:,}
    - Data Period: {data['Year'].min() if 'Year' in data.columns else 'N/A'} - {data['Year'].max() if 'Year' in data.columns else 'N/A'}
    
    🏆 TOP PERFORMERS:
    """
                # Top Vehicle Classes
                if 'Vehicle Class' in data.columns:
                    top_vehicles = data.groupby('Vehicle Class')['TOTAL'].sum().sort_values(ascending=False).head(5)
                    summary_text += "\nTop Vehicle Classes:\n"
                    for vehicle, total in top_vehicles.items():
                        summary_text += f"- {vehicle}: {total:,}\n"
    
                # Top States
                state_col = 'State' if 'State' in data.columns else ('Filter_State' if 'Filter_State' in data.columns else None)
                if state_col:
                    top_states = data.groupby(state_col)['TOTAL'].sum().sort_values(ascending=False).head(5)
                    summary_text += "\nTop States:\n"
                    for state, total in top_states.items():
                        summary_text += f"- {state}: {total:,}\n"
    
                st.download_button(
                    label="💾 Download Summary Report",
                    data=summary_text,
                    file_name=f"vahan_summary_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
                st.success("✅ Summary report ready for download!")
    
        # ----------------------------
        # 3️⃣ Export Info
        # ----------------------------
        with col3:
            st.info(
                "💡 **Export Options**\n\n"
                "You can download the processed dataset for further analysis in Excel or other tools.\n"
                "Additionally, the summary report provides a quick overview of total registrations, "
                "top vehicle classes, and top states for investor insights."
            )

def main():
    """🚀 Main VAHAN Vehicle Registration Analytics Dashboard"""
    st.markdown(
        '<h1 class="main-header">🚗 VAHAN Vehicle Registration Analytics</h1>', 
        unsafe_allow_html=True
    )
    
    # Initialize modular dashboard
    dashboard = ModularVahanDashboard()
    
    # ----------------------------
    # Sidebar: Data Source
    # ----------------------------
    st.sidebar.title("📊 Data Management")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Upload CSV File", "🌐 Live Scraping"]
    )
    
    source_param = "upload" if data_source == "Upload CSV File" else "scrape"
    
    # ----------------------------
    # Load Data
    # ----------------------------
    if dashboard.load_data(data_source=source_param):
        # Sidebar filters
        filters = dashboard.create_sidebar_filters()
        filtered_data = dashboard.apply_filters(filters)
        
        if len(filtered_data) > 0:
            # ----------------------------
            # Dashboard Tabs
            # ----------------------------
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "📈 Growth Analysis", "🥧 Market Share", 
                "🔄 Comparisons", "💡 Insights"
            ])
            
            # 1️⃣ Overview
            with tab1:
                st.header("📊 Dashboard Overview")
                dashboard.create_kpi_cards(filtered_data)
                st.markdown("---")
                dashboard.create_time_series_analysis(filtered_data)
            
            # 2️⃣ Growth
            with tab2:
                dashboard.create_growth_charts(filtered_data)
            
            # 3️⃣ Market Share
            with tab3:
                dashboard.create_market_share_analysis(filtered_data)
                st.markdown("---")
                dashboard.create_manufacturer_analysis(filtered_data)
            
            # 4️⃣ Comparisons
            with tab4:
                dashboard.create_comparison_tool(filtered_data)
            
            # 5️⃣ Investor Insights
            with tab5:
                dashboard.create_investor_insights(filtered_data)
                st.markdown("---")
                dashboard.create_export_section(filtered_data)
            
            # ----------------------------
            # Footer
            # ----------------------------
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; color: #666; font-size: 0.9em;">
            Showing {len(filtered_data):,} records | 
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Data period: {filtered_data['Year'].min() if 'Year' in filtered_data.columns else 'N/A'} - 
            {filtered_data['Year'].max() if 'Year' in filtered_data.columns else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.error("⚠️ No data available with current filters. Adjust your selections in the sidebar.")
    
    else:
        st.info("👆 Load data using the sidebar to begin analysis.")
        st.markdown("""
        ### 📋 Getting Started with Modular VAHAN Analytics
        1. **Data Loading**: Choose from two options:
           - **Upload CSV**: Upload your own VAHAN data file
           - **🌐 Live Scraping**: Scrape real-time data directly from VAHAN website
        2. **Configure Scraping** (if Live Scraping): Select states, years, and data types
        3. **Apply Filters**: Filter by year, state, and vehicle category in the sidebar
        4. **Explore Tabs**: Navigate through Overview, Growth, Market Share, Comparisons, and Investor Insights
        
        ### 🏗️ Modular Dashboard Features
        - **Maxed KPIs**: Total registrations, states, YoY growth, data period
        - **Growth Analysis**: CAGR, category-wise growth, stability analysis
        - **Market Insights**: Market share, top states, top vehicle classes
        - **Comparison Tools**: Interactive state & vehicle comparisons
        - **Investor-Focused Insights**: Market overview, growth signals, top 3 class share
        - **Export Options**: Download processed data & summary report
        - **Enhanced UI**: Maxed titles, emojis, and modular layout
        """)
        
if __name__ == "__main__":
    main()
