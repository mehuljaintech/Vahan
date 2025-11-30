from pathlib import Path
from datetime import datetime

class Config:
    """
    MAXED 2025 Hardened VAHAN Configuration
    ---------------------------------------
    Centralized configuration for web scraping, processing, and logging.
    Features:
        ✓ Base directories auto-created
        ✓ Logging format & level
        ✓ Data export helpers
        ✓ Static dropdown IDs
        ✓ Vehicle categories
        ✓ Numeric columns
        ✓ Fully error-tolerant and maxed
    """

    # ---------- BASE DIRECTORIES ----------
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    LOGS_DIR = BASE_DIR / "logs"

    # ---------- VAHAN WEBSITE ----------
    VAHAN_BASE_URL = "https://vahan.parivahan.gov.in/vahan4dashboard/vahan/view/reportview.xhtml"
    DEFAULT_WAIT_TIME = 15
    MAX_RETRIES = 3

    HEADLESS_MODE = True
    WINDOW_SIZE = "1920,1080"

    # ---------- DATA PROCESSING ----------
    NUMERIC_COLUMNS = ['2WIC', '2WN', '2WT', '3WN', '3WT', 'LMV', 'MMV', 'HMV', 'TOTAL']

    VEHICLE_CATEGORIES = {
        '2W': ['MOTOR CYCLE', 'SCOOTER', 'M-CYCLE', 'MOPED'],
        '3W': ['AUTO RICKSHAW', '3W', 'THREE WHEELER'],
        '4W+': ['CAR', 'TRUCK', 'BUS', 'LMV', 'MMV', 'HMV']
    }

    # ---------- STATIC DROPDOWNS ----------
    STATIC_DROPDOWNS = {
        "Y-Axis": "yaxisVar",
        "X-Axis": "xaxisVar",
        "Year": "selectedYear",
        "Year Type": "selectedYearType",
        "Vehicle Type": "vchgroupTable:selectCatgGrp"
    }

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

    @classmethod
    def get_dropdown_id(cls, dropdown_name: str) -> str:
        """Safe retrieval of static dropdown IDs."""
        try:
            return cls.STATIC_DROPDOWNS.get(dropdown_name, "UNKNOWN")
        except Exception:
            return "UNKNOWN"
