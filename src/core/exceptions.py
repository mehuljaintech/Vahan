"""
MAXED 2025 Hardened Custom Exceptions for VAHAN Web Scraper
-----------------------------------------------------------
Provides specific, traceable, and hierarchical error handling for all scraper operations.
"""

# ---------- BASE EXCEPTION ----------
class VahanScraperError(Exception):
    """Base exception for all VAHAN scraper-related errors."""
    def __init__(self, message="An error occurred in the VAHAN scraper", code=None, **kwargs):
        super().__init__(message)
        self.code = code
        self.extra = kwargs

# ---------- SCRAPING ERRORS ----------
class ScrapingError(VahanScraperError):
    """Raised when web scraping operations fail."""
    def __init__(self, message="Scraping operation failed", **kwargs):
        super().__init__(message, **kwargs)

class DropdownError(ScrapingError):
    """Raised when dropdown interaction fails."""
    def __init__(self, dropdown_name=None, message=None, **kwargs):
        msg = message or f"Dropdown interaction failed for '{dropdown_name}'"
        super().__init__(msg, **kwargs)
        self.dropdown_name = dropdown_name

class DynamicIDError(ScrapingError):
    """Raised when dynamic ID detection fails."""
    def __init__(self, element_name=None, message=None, **kwargs):
        msg = message or f"Dynamic ID detection failed for '{element_name}'"
        super().__init__(msg, **kwargs)
        self.element_name = element_name

# ---------- DATA PROCESSING ERRORS ----------
class DataProcessingError(VahanScraperError):
    """Raised when data processing operations fail."""
    def __init__(self, message="Data processing failed", dataset=None, **kwargs):
        super().__init__(message, **kwargs)
        self.dataset = dataset

class ValidationError(VahanScraperError):
    """Raised when data validation fails."""
    def __init__(self, message="Validation failed", field=None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field

# ---------- CONFIGURATION & EXPORT ----------
class ConfigurationError(VahanScraperError):
    """Raised when configuration is invalid or missing."""
    def __init__(self, message="Invalid configuration", config_name=None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_name = config_name

class ExportError(VahanScraperError):
    """Raised when data export operations fail."""
    def __init__(self, message="Export operation failed", filepath=None, **kwargs):
        super().__init__(message, **kwargs)
        self.filepath = filepath
