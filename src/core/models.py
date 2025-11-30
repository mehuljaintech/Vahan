"""
Data models and type definitions for VAHAN web scraper.
Provides structured data types for better type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

@dataclass
class ScrapingConfig:
    """
    MAXED 2025 Hardened Scraper Configuration
    -----------------------------------------
    Centralized configuration for VAHAN web scraping operations.
    Includes validation, defaults, and safety checks.
    """
    states: List[str]
    years: List[str]
    vehicle_types: Optional[List[str]] = field(default_factory=list)
    headless: bool = True
    wait_time: int = 15
    max_retries: int = 3

    # ---------- MAXED ADDITIONS ----------
    def validate(self):
        """Validate scraping configuration with robust checks."""
        if not self.states or not isinstance(self.states, list):
            raise ValueError("⚠️ States must be a non-empty list")
        if not all(isinstance(s, str) and s.strip() for s in self.states):
            raise ValueError("⚠️ All states must be non-empty strings")
        if not self.years or not isinstance(self.years, list):
            raise ValueError("⚠️ Years must be a non-empty list")
        if not all(isinstance(y, str) and y.strip() for y in self.years):
            raise ValueError("⚠️ All years must be non-empty strings")
        if self.vehicle_types is not None:
            if not isinstance(self.vehicle_types, list):
                raise ValueError("⚠️ Vehicle types must be a list if provided")
            if not all(isinstance(v, str) and v.strip() for v in self.vehicle_types):
                raise ValueError("⚠️ All vehicle types must be non-empty strings")
        if self.wait_time <= 0:
            raise ValueError("⚠️ Wait time must be positive")
        if self.max_retries < 0:
            raise ValueError("⚠️ Max retries cannot be negative")
        return True

@dataclass
class FilterCombination:
    """
    MAXED 2025 Hardened Filter Combination
    --------------------------------------
    Represents a single combination of filter parameters for VAHAN scraping.
    Includes robust validation and safe dict conversion.
    """
    state: str
    year: str
    vehicle_type: Optional[str] = None
    y_axis: Optional[str] = None
    x_axis: Optional[str] = None

    # ---------- MAXED ADDITIONS ----------
    def as_dict(self) -> dict:
        """
        Return the filter combination as a dictionary for Selenium or API usage.
        Ensures all fields are included, even if None.
        """
        return {
            "state": self.state,
            "year": self.year,
            "vehicle_type": self.vehicle_type,
            "y_axis": self.y_axis,
            "x_axis": self.x_axis
        }

    def validate(self) -> bool:
        """
        Validate the filter combination.
        Raises ValueError if any required field is invalid.
        """
        if not self.state or not isinstance(self.state, str) or not self.state.strip():
            raise ValueError("⚠️ State must be a non-empty string")
        if not self.year or not isinstance(self.year, str) or not self.year.strip():
            raise ValueError("⚠️ Year must be a non-empty string")
        if self.vehicle_type is not None:
            if not isinstance(self.vehicle_type, str) or not self.vehicle_type.strip():
                raise ValueError("⚠️ Vehicle type must be a non-empty string if provided")
        return True

@dataclass
class ScrapingResult:
    """
    MAXED 2025 Hardened Scraping Result
    -----------------------------------
    Encapsulates the output of a VAHAN scraping operation.
    Includes data, metadata, timestamps, success flags, and optional error info.
    """
    data: pd.DataFrame
    metadata: Dict
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """
        Quick validation of scraping result.
        Returns True if result is structurally sound.
        """
        if not isinstance(self.data, pd.DataFrame):
            return False
        if self.data.empty and self.success:
            return False
        if not isinstance(self.timestamp, datetime):
            return False
        return True

    def summary(self) -> Dict:
        """
        Return a concise, executive-ready summary of the scraping result.
        """
        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "timestamp": self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "success": self.success,
            "error_message": self.error_message
        }

    def as_dict(self) -> Dict:
        """
        Full dictionary representation including metadata for JSON export or logging.
        """
        return {
            "data_preview": self.data.head(5).to_dict(orient="records"),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message
        }

@dataclass
class GrowthMetrics:
    """
    MAXED 2025 Hardened Growth Metrics
    ----------------------------------
    Encapsulates all growth metrics for market analysis:
    - YoY & QoQ growth
    - Category, state, manufacturer growth
    - Fully nested and type-safe
    """
    yoy_growth: Dict[str, float]
    qoq_growth: Dict[str, float]
    category_growth: Dict[str, Dict[str, float]]
    state_growth: Dict[str, Dict[str, float]]
    manufacturer_growth: Dict[str, Dict[str, float]]

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """
        Validate that all metrics are non-empty and numeric.
        Fully recursive for nested dictionaries.
        """
        def _check_dict(d):
            if not isinstance(d, dict) or not d:
                return False
            for k, v in d.items():
                if isinstance(v, dict):
                    if not _check_dict(v):
                        return False
                else:
                    if not isinstance(v, (int, float)):
                        return False
            return True

        return all([
            _check_dict(self.yoy_growth),
            _check_dict(self.qoq_growth),
            _check_dict(self.category_growth),
            _check_dict(self.state_growth),
            _check_dict(self.manufacturer_growth)
        ])

    def summary(self) -> Dict[str, int]:
        """
        Return counts of available entries per metric type for a quick executive overview.
        """
        return {
            "yoy_entries": len(self.yoy_growth),
            "qoq_entries": len(self.qoq_growth),
            "category_entries": len(self.category_growth),
            "state_entries": len(self.state_growth),
            "manufacturer_entries": len(self.manufacturer_growth)
        }

@dataclass
class MarketInsights:
    """
    MAXED 2025 Hardened Market Insights
    -----------------------------------
    Encapsulates comprehensive market analysis:
    - Market overview
    - Growth leaders
    - Risk factors
    - Investment opportunities
    - Top manufacturers & market share
    """
    market_overview: Dict
    growth_leaders: List[str]
    risk_factors: List[str]
    investment_opportunities: List[str]
    top_manufacturers: List[Dict]
    market_share: Dict[str, float]

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """
        Validate that insights contain meaningful and correctly typed content.
        """
        return all([
            isinstance(self.market_overview, dict) and bool(self.market_overview),
            isinstance(self.growth_leaders, list),
            isinstance(self.risk_factors, list),
            isinstance(self.investment_opportunities, list),
            isinstance(self.top_manufacturers, list),
            isinstance(self.market_share, dict)
        ])

    def summary(self) -> Dict[str, int]:
        """
        Return counts of key insight elements for dashboards or executive overview.
        """
        return {
            "growth_leaders_count": len(self.growth_leaders),
            "risk_factors_count": len(self.risk_factors),
            "investment_opportunities_count": len(self.investment_opportunities),
            "top_manufacturers_count": len(self.top_manufacturers),
            "market_share_entries": len(self.market_share)
        }

@dataclass
class ProcessingResult:
    """
    MAXED 2025 Hardened Data Processing Result
    ------------------------------------------
    Encapsulates outcomes of data cleaning, growth analysis, 
    and market insights generation.
    """
    cleaned_data: pd.DataFrame
    growth_metrics: Any  # GrowthMetrics instance
    insights: Any        # MarketInsights instance
    processing_time: float
    records_processed: int

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """
        Validate processing result integrity:
        - Cleaned data is a DataFrame
        - Growth metrics and insights are valid
        - Processing time and record counts are non-negative
        """
        return all([
            isinstance(self.cleaned_data, pd.DataFrame),
            hasattr(self.growth_metrics, "is_valid") and self.growth_metrics.is_valid(),
            hasattr(self.insights, "is_valid") and self.insights.is_valid(),
            isinstance(self.processing_time, (int, float)) and self.processing_time >= 0,
            isinstance(self.records_processed, int) and self.records_processed >= 0
        ])

    def summary(self) -> Dict[str, Any]:
        """
        Return a concise summary of the processing result:
        - Record counts
        - Cleaned rows and columns
        - Processing duration
        - Validation status of metrics and insights
        """
        return {
            "records_processed": self.records_processed,
            "rows_cleaned": len(self.cleaned_data),
            "columns_cleaned": list(self.cleaned_data.columns),
            "processing_time_sec": round(self.processing_time, 2),
            "growth_metrics_valid": self.growth_metrics.is_valid(),
            "insights_valid": self.insights.is_valid()
        }
