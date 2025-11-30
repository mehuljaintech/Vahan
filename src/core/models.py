"""
Data models and type definitions for VAHAN web scraper.
Provides structured data types for better type safety and validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

@dataclass
class ScrapingConfig:
    """Configuration for scraping operations."""
    states: List[str]
    years: List[str]
    vehicle_types: Optional[List[str]] = None
    headless: bool = True
    wait_time: int = 15
    max_retries: int = 3

    # ---------- MAXED ADDITIONS ----------
    def validate(self):
        """Validate scraping configuration."""
        if not self.states or not isinstance(self.states, list):
            raise ValueError("States must be a non-empty list")
        if not self.years or not isinstance(self.years, list):
            raise ValueError("Years must be a non-empty list")
        if self.wait_time <= 0:
            raise ValueError("Wait time must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

@dataclass
class FilterCombination:
    """Represents a filter combination for scraping."""
    state: str
    year: str
    vehicle_type: Optional[str] = None
    y_axis: Optional[str] = None
    x_axis: Optional[str] = None

    # ---------- MAXED ADDITIONS ----------
    def as_dict(self):
        """Return the filter combination as a dictionary for Selenium or API usage."""
        return {
            "state": self.state,
            "year": self.year,
            "vehicle_type": self.vehicle_type,
            "y_axis": self.y_axis,
            "x_axis": self.x_axis
        }

    def validate(self):
        """Validate filter combination."""
        if not self.state or not isinstance(self.state, str):
            raise ValueError("State must be a non-empty string")
        if not self.year or not isinstance(self.year, str):
            raise ValueError("Year must be a non-empty string")

@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    data: pd.DataFrame
    metadata: Dict
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """Quick validation of scraping result."""
        if not isinstance(self.data, pd.DataFrame):
            return False
        if self.data.empty and self.success:
            return False
        if not isinstance(self.timestamp, datetime):
            return False
        return True

    def summary(self) -> Dict:
        """Return a concise summary of the scraping result."""
        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "timestamp": self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "success": self.success,
            "error_message": self.error_message
        }

@dataclass
class GrowthMetrics:
    """Growth metrics for analysis."""
    yoy_growth: Dict[str, float]
    qoq_growth: Dict[str, float]
    category_growth: Dict[str, Dict[str, float]]
    state_growth: Dict[str, Dict[str, float]]
    manufacturer_growth: Dict[str, Dict[str, float]]

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """Validate that growth metrics are non-empty and numeric."""
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
        """Return counts of available metrics for quick overview."""
        return {
            "yoy_entries": len(self.yoy_growth),
            "qoq_entries": len(self.qoq_growth),
            "category_entries": len(self.category_growth),
            "state_entries": len(self.state_growth),
            "manufacturer_entries": len(self.manufacturer_growth)
        }

@dataclass
class MarketInsights:
    """Market insights and analysis results."""
    market_overview: Dict
    growth_leaders: List[str]
    risk_factors: List[str]
    investment_opportunities: List[str]
    top_manufacturers: List[Dict]
    market_share: Dict[str, float]

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """Validate that the insights have meaningful content."""
        return all([
            isinstance(self.market_overview, dict) and bool(self.market_overview),
            isinstance(self.growth_leaders, list),
            isinstance(self.risk_factors, list),
            isinstance(self.investment_opportunities, list),
            isinstance(self.top_manufacturers, list),
            isinstance(self.market_share, dict)
        ])

    def summary(self) -> Dict[str, int]:
        """Return counts of key insight elements for dashboard/overview."""
        return {
            "growth_leaders_count": len(self.growth_leaders),
            "risk_factors_count": len(self.risk_factors),
            "investment_opportunities_count": len(self.investment_opportunities),
            "top_manufacturers_count": len(self.top_manufacturers),
            "market_share_entries": len(self.market_share)
        }

@dataclass
class ProcessingResult:
    """Result of data processing operations."""
    cleaned_data: pd.DataFrame
    growth_metrics: Any  # GrowthMetrics instance
    insights: Any        # MarketInsights instance
    processing_time: float
    records_processed: int

    # ---------- MAXED ADDITIONS ----------
    def is_valid(self) -> bool:
        """Validate processing result integrity."""
        return all([
            isinstance(self.cleaned_data, pd.DataFrame),
            hasattr(self.growth_metrics, "is_valid") and self.growth_metrics.is_valid(),
            hasattr(self.insights, "is_valid") and self.insights.is_valid(),
            isinstance(self.processing_time, (int, float)) and self.processing_time >= 0,
            isinstance(self.records_processed, int) and self.records_processed >= 0
        ])

    def summary(self) -> dict:
        """Return a concise summary of the processing result."""
        return {
            "records_processed": self.records_processed,
            "rows_cleaned": len(self.cleaned_data),
            "columns_cleaned": list(self.cleaned_data.columns),
            "processing_time_sec": round(self.processing_time, 2),
            "growth_metrics_valid": self.growth_metrics.is_valid(),
            "insights_valid": self.insights.is_valid()
        }
