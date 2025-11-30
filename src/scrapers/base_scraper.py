"""
Abstract base scraper for web scraping operations.
Provides common functionality and interface for all scrapers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

from ..core.config import Config
from ..core.exceptions import ScrapingError, ConfigurationError
from ..utils.logging_utils import get_logger

class BaseScraper(ABC):
    """Abstract base class for web scrapers."""
    
    def __init__(self, base_url: str, wait_time: int = None):
        """
        MAXED 2025 EDITION
        ------------------
        Initialize the base web scraper with Chrome WebDriver, wait configuration, and logging.
    
        Args:
            base_url (str): The base URL to scrape
            wait_time (int, optional): Maximum wait time for page elements (defaults to Config.DEFAULT_WAIT_TIME)
        """
        from typing import Optional
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
    
        self.base_url = base_url
        self.wait_time: int = wait_time or Config.DEFAULT_WAIT_TIME
    
        # Selenium WebDriver placeholders
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
    
        # Initialize logger
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug(f"Scraper initialized for {self.base_url} with wait_time={self.wait_time}s")
        
    def setup_driver(self, headless: bool = None) -> None:
        """
        MAXED 2025 EDITION
        ------------------
        Initialize Chrome WebDriver with robust options, stealth configuration, 
        and wait management for scraping.
    
        Args:
            headless (bool, optional): Whether to run in headless mode (defaults to Config.HEADLESS_MODE)
        """
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
    
        # Determine headless mode
        headless = Config.HEADLESS_MODE if headless is None else headless
    
        # --- Step 1: Configure Chrome options ---
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
    
        # Performance & compatibility options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={Config.WINDOW_SIZE}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
    
        # --- Step 2: Initialize WebDriver safely ---
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.wait = WebDriverWait(self.driver, self.wait_time)
            self.logger.info(f"✅ WebDriver initialized successfully (headless={headless})")
        except Exception as e:
            self.logger.error(f"❌ WebDriver initialization failed: {e}")
            raise ConfigurationError(f"Failed to initialize WebDriver: {e}")
    
    def open_page(self) -> None:
        """
        MAXED 2025 EDITION
        ------------------
        Open the target page with enhanced error handling, logging, 
        and ready for dynamic content scraping.
    
        Raises:
            ScrapingError: If WebDriver is not initialized or page fails to load
        """
        from selenium.common.exceptions import WebDriverException, TimeoutException
    
        if not self.driver:
            self.logger.error("❌ WebDriver not initialized. Call setup_driver() first.")
            raise ScrapingError("WebDriver not initialized. Call setup_driver() first.")
    
        try:
            self.driver.get(self.base_url)
            self.logger.info(f"📖 Opened page successfully: {self.base_url}")
    
            # Optional: Wait until the page is fully loaded
            self.driver.execute_script("return document.readyState === 'complete'")
            self.logger.debug("✅ Page load verified (readyState complete)")
    
        except (WebDriverException, TimeoutException) as e:
            self.logger.error(f"❌ Failed to open page {self.base_url}: {e}")
            raise ScrapingError(f"Failed to open page: {e}")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error while opening page: {e}")
            raise ScrapingError(f"Unexpected error opening page: {e}")
    
    def close(self) -> None:
        """
        MAXED 2025 EDITION
        ------------------
        Close the browser, quit the WebDriver, and clean up resources
        safely, with detailed logging and exception handling.
        """
        from selenium.common.exceptions import WebDriverException
    
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("🔄 Browser closed successfully and resources released")
            except WebDriverException as e:
                self.logger.warning(f"⚠️ WebDriver quit failed: {e}")
            except Exception as e:
                self.logger.warning(f"⚠️ Unexpected error during browser close: {e}")
            finally:
                self.driver = None
                self.wait = None
                self.logger.debug("✅ Internal driver and wait references cleared")
        else:
            self.logger.debug("ℹ️ No WebDriver instance found; nothing to close")
    
    from abc import abstractmethod
    import pandas as pd
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        MAXED 2025 EDITION
        ------------------
        Abstract method to fetch data from the website.
    
        Implementing subclasses must:
        - Handle all network/browser interactions
        - Ensure proper data extraction into a pandas DataFrame
        - Include logging for success/failure
        - Raise ScrapingError if the fetch fails
    
        Returns:
            pd.DataFrame: Scraped and structured data
    
        Raises:
            ScrapingError: If scraping fails or data cannot be returned
        """
        pass
    
    from abc import abstractmethod
    from typing import Dict
    
    @abstractmethod
    def apply_filters(self, filters: Dict[str, str]) -> None:
        """
        MAXED 2025 EDITION
        ------------------
        Abstract method to apply filters on the target website.
    
        Implementing subclasses must:
        - Interact with the page to set all relevant filters
        - Validate that filters are applied successfully
        - Handle dynamic elements, dropdowns, checkboxes, or date pickers
        - Include logging for each filter applied
        - Raise ScrapingError if any filter cannot be applied
    
        Args:
            filters (Dict[str, str]): Mapping of filter names to their desired values
    
        Raises:
            ScrapingError: If filters cannot be applied or verified
        """
        pass
    
    from abc import abstractmethod
    from typing import Dict, List
    
    @abstractmethod
    def scrape_dropdowns(self) -> Dict[str, List[str]]:
        """
        MAXED 2025 EDITION
        ------------------
        Abstract method to scrape all available dropdown options from the target website.
    
        Implementing subclasses must:
        - Identify all relevant dropdown elements dynamically
        - Extract visible option text for each dropdown
        - Handle lazy-loaded or dynamic dropdowns that require scrolling or clicks
        - Include robust error handling for missing or empty dropdowns
        - Log the number of options scraped per dropdown
        - Return a complete mapping of dropdown names to option lists
    
        Returns:
            Dict[str, List[str]]: Mapping of dropdown element names to their available options
    
        Raises:
            ScrapingError: If dropdowns cannot be located, scraped, or validated
        """
        pass
    
    def __enter__(self):
        """
        MAXED 2025 CONTEXT MANAGER ENTRY
    
        Allows usage of the scraper within a `with` statement.
        Ensures the instance is returned properly and is ready for operations.
        
        Example:
            with MyScraper(base_url) as scraper:
                scraper.setup_driver()
                scraper.open_page()
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        MAXED 2025 CONTEXT MANAGER EXIT
    
        Ensures proper cleanup of the scraper resources when exiting a `with` block.
        Automatically closes the WebDriver even if exceptions occur.
    
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Traceback (if any)
        """
        self.close()
