"""
VAHAN-specific scraper implementation.
Handles PrimeFaces dropdown interactions with dynamic IDs.
"""

import time
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper
from ..core.config import Config
from ..core.exceptions import ScrapingError, DropdownError, DynamicIDError
from ..core.models import FilterCombination, ScrapingResult
from ..utils.logging_utils import get_logger

class VahanScraper(BaseScraper):
    """Enhanced scraper for VAHAN dashboard that handles PrimeFaces dropdown interactions with dynamic IDs."""
    
    def __init__(self, base_url: str = None, wait_time: int = None):
        """
        MAXED 2025 VAHAN SCRAPER INIT
    
        Initialize the VAHAN scraper with logging, WebDriver setup placeholders,
        and storage for scraped results.
    
        Args:
            base_url (str, optional): VAHAN dashboard URL. Defaults to Config.VAHAN_BASE_URL.
            wait_time (int, optional): Maximum wait time for page elements. Defaults to Config.DEFAULT_WAIT_TIME.
        """
        super().__init__(
            base_url or Config.VAHAN_BASE_URL,
            wait_time or Config.DEFAULT_WAIT_TIME
        )
        self.dynamic_state_id: Optional[str] = None
        self.dynamic_refresh_id: Optional[str] = None
        self.scraped_data: List[Dict] = []
        
    def open_page(self) -> None:
        """
        MAXED 2025 VAHAN PAGE OPEN
    
        Open the VAHAN dashboard using the base scraper logic, wait for full load,
        and detect dynamic element IDs required for interactions.
        """
        super().open_page()
        import time
        time.sleep(3)  # Ensure page is fully loaded
        self._detect_dynamic_ids()
        
    def _detect_dynamic_ids(self) -> None:
        """
        MAXED 2025 VAHAN DYNAMIC IDS
    
        Detect the dynamic IDs for state dropdown and refresh button on the dashboard.
        This includes multiple fallbacks to handle UI changes.
        """
        try:
            # --- Detect State Dropdown ---
            state_selects = self.driver.find_elements(
                By.XPATH,
                "//select[option[contains(@data-escape, 'true') and "
                "(contains(text(), 'Karnataka') or contains(text(), 'Delhi') or contains(text(), 'Maharashtra'))]]"
            )
    
            if state_selects:
                state_select = state_selects[0]
                state_select_id = state_select.get_attribute('id')
                if state_select_id and state_select_id.endswith('_input'):
                    self.dynamic_state_id = state_select_id[:-6]
                    self.logger.info(f"✓ Detected state dropdown ID: {self.dynamic_state_id}")
                else:
                    parent_div = state_select.find_element(By.XPATH, "..")
                    while parent_div and not parent_div.get_attribute('class').startswith('ui-selectonemenu'):
                        parent_div = parent_div.find_element(By.XPATH, "..")
                    if parent_div:
                        self.dynamic_state_id = parent_div.get_attribute('id')
                        self.logger.info(f"✓ Detected state dropdown ID: {self.dynamic_state_id}")
    
            if not self.dynamic_state_id:
                state_divs = self.driver.find_elements(
                    By.XPATH,
                    "//div[contains(@class, 'ui-selectonemenu')]/label[contains(text(), 'States') or contains(text(), 'Vahan4')]/.."
                )
                if state_divs:
                    self.dynamic_state_id = state_divs[0].get_attribute('id')
                    self.logger.info(f"✓ Detected state dropdown ID (fallback): {self.dynamic_state_id}")
    
            # --- Detect Refresh Button ---
            refresh_buttons = self.driver.find_elements(
                By.XPATH,
                "//button[contains(@class, 'ui-button') and (contains(text(), 'Refresh') "
                "or contains(@title, 'Refresh') or .//span[contains(@class, 'ui-icon-refresh')])]"
            )
            if refresh_buttons:
                self.dynamic_refresh_id = refresh_buttons[0].get_attribute('id')
                self.logger.info(f"✓ Detected refresh button ID: {self.dynamic_refresh_id}")
            else:
                refresh_icons = self.driver.find_elements(
                    By.XPATH,
                    "//span[contains(@class, 'ui-icon-refresh')]/parent::button"
                )
                if refresh_icons:
                    self.dynamic_refresh_id = refresh_icons[0].get_attribute('id')
                    self.logger.info(f"✓ Detected refresh button ID (fallback): {self.dynamic_refresh_id}")
    
            # --- Final validation ---
            if not self.dynamic_state_id:
                raise DynamicIDError("❌ Could not detect dynamic state dropdown ID")
            if not self.dynamic_refresh_id:
                self.logger.warning("⚠️ Could not detect refresh button ID - refresh may not work")
    
        except Exception as e:
            raise DynamicIDError(f"❌ Failed to detect dynamic IDs: {e}")
    
    @property
    def dropdowns(self) -> Dict[str, str]:
        """
        MAXED 2025 VAHAN DROPDOWNS
    
        Get the mapping of dropdowns including dynamically detected state ID.
        Combines static config dropdowns with live dynamic IDs detected from the page.
        """
        dropdown_map = Config.STATIC_DROPDOWNS.copy()
        if self.dynamic_state_id:
            dropdown_map["State"] = self.dynamic_state_id
            self.logger.debug(f"✅ Dynamic state dropdown included: {self.dynamic_state_id}")
        else:
            self.logger.debug("⚠️ Dynamic state dropdown not detected, using static config")
        return dropdown_map
    
    def fetch_data(self) -> Dict:
        """
        MAXED 2025 VAHAN TABLE SCRAPING
    
        Fetch and parse data from the VAHAN dashboard table with robust handling,
        logging, and partial row preview for debugging.
    
        Returns:
            dict: Contains headers, rows, status, and total row count
        """
        try:
            # Close any open panels to avoid DOM conflicts
            self._close_all_open_panels()
    
            # Wait for table to be visible
            self.wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "#combTablePnl table"))
            )
    
            # Parse complex table headers (multi-row headers supported)
            headers = self._parse_complex_table_headers()
    
            # Collect table rows
            data = []
            row_elements = self.driver.find_elements(By.CSS_SELECTOR, "#combTablePnl tbody tr")
            for row in row_elements:
                cells = [cell.text.strip() for cell in row.find_elements(By.TAG_NAME, "td")]
                if any(cells):
                    data.append(cells)
    
            # Logging summary
            self.logger.info(f"📊 Table fetched - Headers: {len(headers)}, Rows: {len(data)}")
            self.logger.debug(f"Headers: {headers}")
            for i, row in enumerate(data[:5]):
                self.logger.debug(f"Row {i+1}: {row}")
            if len(data) > 5:
                self.logger.debug(f"... and {len(data) - 5} more rows")
    
            return {
                "headers": headers,
                "rows": data,
                "status": "success",
                "total_rows": len(data)
            }
    
        except TimeoutException:
            self.logger.warning("⚠️ Timeout waiting for table to load")
            return {"headers": [], "rows": [], "status": "timeout"}
    
        except Exception as e:
            self.logger.error(f"⚠️ Error fetching data: {e}")
            return {"headers": [], "rows": [], "status": "error", "error": str(e)}
    
    def _parse_complex_table_headers(self) -> List[str]:
        """
        MAXED 2025: Parse complex VAHAN table headers with robust colspan/rowspan support.
        
        Handles multi-row headers, dynamic categories, standardizes known vehicle types, 
        and falls back safely to simple headers if parsing fails.
        
        Returns:
            List[str]: Final ordered list of header names
        """
        try:
            # Get table HTML and parse with BeautifulSoup
            table_element = self.driver.find_element(By.CSS_SELECTOR, "#combTablePnl table")
            table_html = table_element.get_attribute('outerHTML')
            soup = BeautifulSoup(table_html, 'html.parser')
    
            # Find thead, with optional id fallback
            thead = soup.find('thead', id='vchgroupTable_head') or soup.find('thead')
            if not thead:
                self.logger.warning("⚠️ No thead found, using simple header fallback")
                return self._get_simple_headers()
    
            header_rows = thead.find_all('tr', role='row') or thead.find_all('tr')
            self.logger.debug(f"Found {len(header_rows)} header rows")
    
            final_headers: List[str] = []
    
            # Multi-row header logic
            if len(header_rows) >= 3:
                # Include first 2 static columns if present
                first_row_cells = header_rows[0].find_all('th')
                for cell in first_row_cells[:2]:
                    text = re.sub(r'\s+', ' ', cell.get_text(strip=True))
                    if 'S No' in text:
                        final_headers.append('S No')
                    elif 'Vehicle Class' in text:
                        final_headers.append('Vehicle Class')
    
                # Dynamic vehicle categories from last header row
                last_row_cells = header_rows[-1].find_all('th')
                for cell in last_row_cells:
                    text = re.sub(r'\s+', ' ', cell.get_text(strip=True))
                    if text and text not in ['S No', 'Vehicle Class', '']:
                        # Normalize known vehicle types
                        if text in ['2WIC', '2WN', '2WT', 'TOTAL', '3WN', '3WT', 'LMV', 'MMV', 'HMV', 'LGV', 'MGV', 'HGV']:
                            final_headers.append(text.strip())
                        elif 'TOTAL' in text.upper():
                            final_headers.append('TOTAL')
                        elif len(text) <= 10:
                            final_headers.append(text)
    
            else:
                # Fallback: single or double row headers
                if header_rows:
                    last_header_row = header_rows[-1]
                    header_cells = last_header_row.find_all(['th', 'td'])
                    for cell in header_cells:
                        text = re.sub(r'\s+', ' ', cell.get_text(strip=True))
                        if text and text not in ['', ' ']:
                            if 'S No' in text:
                                final_headers.append('S No')
                            elif 'Vehicle Class' in text:
                                final_headers.append('Vehicle Class')
                            elif text in ['2WIC', '2WN', '2WT', 'TOTAL', '3WN', '3WT', 'LMV', 'MMV', 'HMV']:
                                final_headers.append(text.strip())
                            else:
                                final_headers.append(text)
    
            self.logger.info(f"✓ Parsed complex headers ({len(final_headers)} columns): {final_headers}")
            return final_headers
    
        except Exception as e:
            self.logger.error(f"⚠️ Error parsing complex headers: {e}")
            return self._get_simple_headers()
    
    def _get_simple_headers(self) -> List[str]:
        """
        MAXED 2025: Fallback header extraction for VAHAN table.
    
        Attempts robust simple header parsing when complex parsing fails.
        Ensures unique, stripped headers. Provides sane defaults if all fails.
    
        Returns:
            List[str]: Ordered list of header names
        """
        try:
            header_elements = self.driver.find_elements(By.CSS_SELECTOR, "#combTablePnl thead th")
            headers: List[str] = []
            for h in header_elements:
                text = re.sub(r'\s+', ' ', h.text.strip())
                if text and text not in headers:
                    headers.append(text)
            if not headers:
                # Fallback default headers if table has no detectable headers
                headers = ['S No', 'Vehicle Class', 'Category 1', 'Category 2', 'Category 3', 'TOTAL']
            self.logger.debug(f"✅ Simple headers parsed: {headers}")
            return headers
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get simple headers: {e}. Using default fallback.")
            return ['S No', 'Vehicle Class', 'Category 1', 'Category 2', 'Category 3', 'TOTAL']
    
    def scrape_dropdowns(self) -> Dict[str, List[str]]:
        """
        MAXED 2025: Scrape all dropdowns from VAHAN dashboard with robust logging.
    
        Returns:
            Dict[str, List[str]]: Mapping of dropdown labels to their visible options
        """
        dropdown_data: Dict[str, List[str]] = {}
        
        for label, dropdown_id in self.dropdowns.items():
            try:
                self.logger.info(f"🔍 Scraping '{label}' dropdown (ID: {dropdown_id})...")
                items: List[str] = self._fetch_one_menu_items(dropdown_id)
                dropdown_data[label] = items
                self.logger.debug(f"✅ Found {len(items)} items in '{label}' dropdown")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to scrape '{label}' dropdown: {e}")
                dropdown_data[label] = []
        
        self.logger.info(f"✅ Completed scraping {len(dropdown_data)} dropdowns")
        return dropdown_data
    
    def _fetch_one_menu_items(self, base_id: str, max_retries: int = 3) -> List[str]:
        """
        MAXED 2025: Fetch items from a PrimeFaces selectOneMenu with robust retry and logging.
    
        Args:
            base_id (str): Base ID of the dropdown (without '_input')
            max_retries (int): Maximum retry attempts for robustness
    
        Returns:
            List[str]: List of dropdown items (or warning messages on failure)
        """
        if base_id.endswith("_input"):
            base_id = base_id[:-6]
    
        for attempt in range(1, max_retries + 1):
            try:
                self._close_all_open_panels()
                if attempt > 1:
                    time.sleep(2)  # backoff between retries
    
                self.logger.debug(f"Attempt {attempt}: Processing dropdown '{base_id}'")
                dropdown = self.wait.until(EC.presence_of_element_located((By.ID, base_id)))
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", dropdown)
                time.sleep(0.5)
    
                # Attempt click
                try:
                    self.wait.until(EC.element_to_be_clickable((By.ID, base_id)))
                    dropdown.click()
                except ElementClickInterceptedException:
                    if attempt == max_retries:
                        raise Exception(f"Click intercepted after {max_retries} attempts")
                    continue
    
                # Wait for panel to appear
                panel = self.wait.until(EC.visibility_of_element_located((By.ID, f"{base_id}_panel")))
                li_elements = panel.find_elements(By.CSS_SELECTOR, "li.ui-selectonemenu-item")
    
                items: List[str] = []
                for li in li_elements:
                    text = li.text.strip()
                    if text and text not in items:
                        items.append(text)
    
                # Close dropdown safely
                try:
                    dropdown.click()
                except:
                    self.driver.find_element(By.TAG_NAME, "body").click()
    
                if items:
                    self.logger.info(f"✅ Found {len(items)} items for '{base_id}'")
                    return items
                else:
                    self.logger.warning(f"⚠️ Dropdown '{base_id}' returned no items on attempt {attempt}")
    
            except TimeoutException as e:
                self.logger.warning(f"⚠️ Attempt {attempt} timeout for '{base_id}': {e}")
            except Exception as e:
                self.logger.warning(f"⚠️ Attempt {attempt} failed for '{base_id}': {str(e)[:150]}...")
    
        return [f"⚠️ Failed after {max_retries} attempts for '{base_id}'"]
    
    def _close_all_open_panels(self) -> None:
        """
        MAXED 2025: Close any open PrimeFaces selectOneMenu panels to prevent click interception.
        """
        try:
            open_panels = self.driver.find_elements(
                By.CSS_SELECTOR,
                "div[id$='_panel'].ui-selectonemenu-panel[style*='display: block'], "
                "div[id$='_panel'].ui-selectonemenu-panel:not([style*='display: none'])"
            )
            for panel in open_panels:
                try:
                    body = self.driver.find_element(By.TAG_NAME, "body")
                    ActionChains(self.driver).move_to_element(body).click().perform()
                    time.sleep(0.5)
                except Exception:
                    continue
        except Exception:
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.click()
                time.sleep(1)
            except Exception:
                pass
    
    def apply_filters(self, filters: dict[str, str]) -> dict:
        """
        MAXED 2025: Apply filters to VAHAN dashboard dropdowns and refresh the table.
        
        Args:
            filters: Dictionary mapping filter labels to desired values.
        
        Returns:
            dict: Result of refreshing the table after filters are applied.
        """
        dropdown_map = self.dropdowns
        
        for label, value in filters.items():
            if label not in dropdown_map:
                self.logger.warning(f"⚠️ Unknown filter label: {label}")
                continue
                
            widget_id = dropdown_map[label]
            dropdown_css = f"{widget_id}"
            panel_css = f"{widget_id}_panel"
            item_xpath = f".//li[contains(@class,'ui-selectonemenu-item') and normalize-space(text())='{value}']"
            
            try:
                self._close_all_open_panels()
                
                dropdown_div = self.wait.until(
                    EC.element_to_be_clickable((By.ID, dropdown_css))
                )
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});",
                    dropdown_div
                )
                dropdown_div.click()
                
                panel = self.wait.until(
                    EC.visibility_of_element_located((By.ID, panel_css))
                )
                
                option = panel.find_element(By.XPATH, item_xpath)
                option.click()
                
                self.logger.info(f"✅ Selected {label}: {value}")
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error applying filter {label}: {e}")
        
        return self._click_refresh_button()
    
    def _select_dropdown_option(self, dropdown_id: str, option_value: str) -> None:
        """
        MAXED 2025: Select an option from a PrimeFaces-style dropdown safely.
    
        Args:
            dropdown_id: The dynamic ID of the dropdown container.
            option_value: The visible text of the option to select.
    
        Raises:
            DropdownError: If selection fails after retries.
        """
        try:
            self._close_all_open_panels()
    
            # Click the dropdown trigger
            trigger_xpath = f"//div[@id='{dropdown_id}']//div[contains(@class, 'ui-selectonemenu-trigger')]"
            trigger = self.wait.until(EC.element_to_be_clickable((By.XPATH, trigger_xpath)))
            ActionChains(self.driver).move_to_element(trigger).click().perform()
            time.sleep(1)
    
            # Locate the desired option and click
            option_xpath = f"//div[@id='{dropdown_id}_panel']//li[@data-label='{option_value}' or normalize-space(text())='{option_value}']"
            option = self.wait.until(EC.element_to_be_clickable((By.XPATH, option_xpath)))
            ActionChains(self.driver).move_to_element(option).click().perform()
    
            self.logger.info(f"✅ Selected '{option_value}' from {dropdown_id}")
    
        except Exception as e:
            raise DropdownError(f"⚠️ Failed to select '{option_value}' from {dropdown_id}: {e}")
    
    def _click_refresh_button(self) -> dict:
        """
        MAXED 2025: Skip actual refresh button and safely fetch VAHAN table data.
    
        Returns:
            dict: Table data with headers, rows, and status.
        """
        try:
            self.logger.info("⏳ Skipping refresh button, fetching data directly...")
            time.sleep(2)  # Allow filter changes to propagate
    
            # Wait for the main table panel to be present
            self.wait.until(EC.presence_of_element_located((By.ID, "combTablePnl")))
    
            # Fetch and return table data
            return self.fetch_data()
    
        except TimeoutException:
            self.logger.warning("⚠️ Table did not load within the timeout period")
            return {"headers": [], "rows": [], "status": "timeout"}
    
        except Exception as e:
            self.logger.error(f"❌ Error fetching table data: {e}")
            return {"headers": [], "rows": [], "status": "error", "error": str(e)}
    
    def scrape_multiple_combinations(self, combinations: list[dict]) -> pd.DataFrame:
        """
        MAXED 2025: Scrape VAHAN dashboard for multiple filter combinations.
    
        Args:
            combinations (list[dict]): List of filter dictionaries
    
        Returns:
            pd.DataFrame: Aggregated data with filters and scrape timestamp
        """
        from datetime import datetime
        import pandas as pd
        import time
    
        all_data = []
    
        for i, filters in enumerate(combinations):
            self.logger.info(f"--- Scraping combination {i+1}/{len(combinations)} --- Filters: {filters}")
    
            try:
                result = self.apply_filters(filters)
    
                if result.get("status") == "success" and result.get("rows"):
                    headers = result.get("headers", [])
                    for row_data in result["rows"]:
                        if len(row_data) >= len(headers):
                            row_dict = dict(zip(headers, row_data))
                            # Add filter metadata
                            for k, v in filters.items():
                                row_dict[f"Filter_{k}"] = v
                            row_dict["Scraped_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            all_data.append(row_dict)
    
                    self.logger.info(f"✅ Scraped {len(result.get('rows', []))} rows for combination {i+1}")
    
                else:
                    self.logger.warning(f"⚠️ No data returned for combination {i+1}")
    
            except Exception as e:
                self.logger.error(f"❌ Error scraping combination {filters}: {e}")
                continue
    
            # Optional delay to avoid overwhelming the server
            time.sleep(2)
    
        if all_data:
            df = pd.DataFrame(all_data)
            self.logger.info(f"📊 Total scraped rows: {len(df)}, columns: {len(df.columns)}")
            return df
        else:
            self.logger.warning("⚠️ No data collected across all combinations")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        MAXED 2025: Save scraped VAHAN data to CSV with full logging.
    
        Args:
            data (pd.DataFrame): Data to save
            filename (str, optional): Desired filename (auto-generated if None)
    
        Returns:
            str: Path to saved CSV
        """
        import pandas as pd
    
        if filename is None:
            filename = Config.get_output_filename("vahan_data")
    
        filepath = Config.OUTPUT_DIR / filename
        Config.ensure_directories()
    
        try:
            data.to_csv(filepath, index=False)
            self.logger.info(f"💾 Data successfully saved to {filepath} ({len(data)} rows, {len(data.columns)} columns)")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"❌ Failed to save data to {filepath}: {e}")
            raise ScrapingError(f"Failed to save data: {e}")
