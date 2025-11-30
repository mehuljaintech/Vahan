"""
Configuration settings for VAHAN web scraper.
Centralized configuration management for all modules.
"""

import os
from pathlib import Path
from typing import Dict, List

class Config:
    """Centralized configuration for VAHAN web scraper."""
    
    """
    ULTRA-MAXED 2025 Hardened Configuration for VAHAN Web Scraper
    -------------------------------------------------------------
    Centralizes all scraper settings, directories, and operational flags.
    Features:
        ✓ Safe paths & auto-creation
        ✓ Headless / GUI toggle
        ✓ Retry & timeout policies
        ✓ Logging & output management
        ✓ Fully traceable & environment-aware
    """

    # ---------- BASE DIRECTORIES ----------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    LOGS_DIR = BASE_DIR / "logs"

    # Ensure directories exist
    for path in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    # ---------- VAHAN WEBSITE SETTINGS ----------
    VAHAN_BASE_URL = "https://vahan.parivahan.gov.in/vahan4dashboard/vahan/view/reportview.xhtml"
    DEFAULT_WAIT_TIME = 15      # seconds to wait for elements to load
    MAX_RETRIES = 3             # maximum retry attempts for network/load failures

    # ---------- BROWSER / SCRAPING SETTINGS ----------
    HEADLESS_MODE = True        # True = no GUI, False = visible browser
    WINDOW_SIZE = "1920,1080"   # browser viewport size
    PAGE_LOAD_TIMEOUT = 30       # maximum time to wait for page to load
    IMPLICIT_WAIT = 10           # default Selenium implicit wait
    SCROLL_DELAY = 0.5           # delay for scrolling actions

    # ---------- LOGGING SETTINGS ----------
    LOG_FILE = LOGS_DIR / "vahan_scraper.log"
    LOG_LEVEL = "INFO"           # DEBUG | INFO | WARNING | ERROR

    # ---------- DATA EXPORT SETTINGS ----------
    DEFAULT_FILE_FORMAT = "xlsx"  # "csv" | "xlsx"
    AUTO_TIMESTAMP = True          # append timestamp to output files

    # ---------- SAFETY & MAXED FLAGS ----------
    SAFE_MODE = True               # prevents destructive actions
    MAX_CACHE_ENTRIES = 20         # internal cache for fetched pages
    RETRY_BACKOFF_FACTOR = 2       # exponential backoff for retries
    
 # ---------- NUMERIC DATA COLUMNS ----------
    NUMERIC_COLUMNS = [
        '2WIC', '2WN', '2WT',
        '3WN', '3WT',
        'LMV', 'MMV', 'HMV',
        'TOTAL'
    ]

    # ---------- VEHICLE CATEGORY MAPPING ----------
    VEHICLE_CATEGORIES = {
        '2W': ['MOTOR CYCLE', 'SCOOTER', 'M-CYCLE', 'MOPED'],
        '3W': ['AUTO RICKSHAW', '3W', 'THREE WHEELER'],
        '4W+': ['CAR', 'TRUCK', 'BUS', 'LMV', 'MMV', 'HMV']
    }

    @classmethod
    def normalize_vehicle_category(cls, vehicle_name: str) -> str:
        """
        Maps any vehicle name to a standard category.
        Returns:
            '2W' | '3W' | '4W+' | 'UNKNOWN'
        """
        try:
            if not vehicle_name or not isinstance(vehicle_name, str):
                return 'UNKNOWN'

            vehicle_name_upper = vehicle_name.strip().upper()
            for cat, keywords in cls.VEHICLE_CATEGORIES.items():
                if vehicle_name_upper in keywords:
                    return cat
            return 'UNKNOWN'
        except Exception:
            return 'UNKNOWN'

    @classmethod
    def validate_numeric_columns(cls, df) -> list:
        """
        Returns a list of columns that exist in df and are numeric-ready.
        """
        existing_cols = [col for col in cls.NUMERIC_COLUMNS if col in df.columns]
        return existing_cols    

    STATIC_DROPDOWNS = {
        "Y-Axis": "yaxisVar",
        "X-Axis": "xaxisVar",
        "Year": "selectedYear",
        "Year Type": "selectedYearType",
        "Vehicle Type": "vchgroupTable:selectCatgGrp"
    }

    @classmethod
    def get_dropdown_id(cls, dropdown_name: str) -> str:
        """
        Returns the dropdown ID for a given human-readable name.
        Returns 'UNKNOWN' if the dropdown name is not found.
        """
        try:
            return cls.STATIC_DROPDOWNS.get(dropdown_name, "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    
  # ---------- LOGGING ----------
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'

    # ---------- EXPORT ----------
    DEFAULT_EXPORT_FORMAT = 'csv'
    TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

    # ---------- HELPERS ----------
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_output_filename(cls, prefix: str, extension: str = None) -> str:
        """Generate timestamped output filename."""
        ext = extension or cls.DEFAULT_EXPORT_FORMAT
        timestamp = datetime.now().strftime(cls.TIMESTAMP_FORMAT)
        return f"{prefix}_{timestamp}.{ext}"
