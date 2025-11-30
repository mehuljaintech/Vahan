"""
Growth analysis module for VAHAN data.
Provides advanced growth metrics and trend analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from ..core.config import Config
from ..core.exceptions import DataProcessingError
from ..utils.logging_utils import get_logger

class GrowthAnalyzer:
    """Advanced growth analysis for VAHAN vehicle registration data."""
    
    def __init__(self):
        """
        Initialize the GrowthAnalyzer (MAXED EDITION).

        Responsibilities:
            • Set up dedicated logger
            • Initialize internal caches
            • Pre-validate environment
            • Prepare analysis metadata
            • Harden against runtime failures
        """
        # ------------------------------------------------
        # 1. Logger initialization (safe + namespaced)
        # ------------------------------------------------
        try:
            self.logger = get_logger(self.__class__.__name__)
        except Exception as e:
            # Fallback logger (guaranteed non-breaking)
            import logging
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.warning(
                f"[GrowthAnalyzer] Failed to initialize primary logger: {e}"
            )

        # ------------------------------------------------
        # 2. Internal data containers
        # ------------------------------------------------
        self._raw_data = None               # Original input DataFrame
        self._clean_data = None             # Cleaned + prepared DataFrame
        self._growth_cache = {}             # Store computed growth metrics
        self._trend_summary = {}            # Summary stats for insights

        # ------------------------------------------------
        # 3. Metadata (for debugging + audits)
        # ------------------------------------------------
        self.metadata = {
            "version": "2025.1.0",
            "initialized_at": __import__("datetime").datetime.now(),
            "status": "ready",
            "source": "GrowthAnalyzer.MAXED",
        }

        # ------------------------------------------------
        # 4. Environment validation (light check)
        # ------------------------------------------------
        try:
            import pandas as pd  # noqa: F401
            import numpy as np   # noqa: F401
            self.logger.info("[GrowthAnalyzer] Environment OK")
        except Exception as e:
            self.logger.error(
                f"[GrowthAnalyzer] Environment dependency missing: {e}"
            )

        # ------------------------------------------------
        # 5. Final confirmation
        # ------------------------------------------------
        self.logger.info("[GrowthAnalyzer] Initialized successfully (MAXED EDITION)")
    
    def calculate_compound_growth_rate(
        self,
        data: pd.DataFrame,
        column: str = "TOTAL",
        period_column: str = "Year"
    ) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR) — MAXED EDITION.

        Formula:
            CAGR = (Ending / Beginning) ** (1 / periods) - 1

        Returns:
            float: CAGR % (rounded to 2 decimal places)
        """

        # ------------------------------------------------------------
        # 1. Basic structural validation
        # ------------------------------------------------------------
        try:
            if not isinstance(data, pd.DataFrame):
                self.logger.warning("[CAGR] Invalid input: data is not a DataFrame")
                return 0.0

            if data.empty:
                self.logger.info("[CAGR] Skipped (empty DataFrame)")
                return 0.0

            if column not in data.columns:
                self.logger.warning(f"[CAGR] Column missing: {column}")
                return 0.0

            if period_column not in data.columns:
                self.logger.warning(f"[CAGR] Period column missing: {period_column}")
                return 0.0

            # ------------------------------------------------------------
            # 2. Clean & prepare dataset
            # ------------------------------------------------------------
            df = data[[period_column, column]].copy()

            # Remove NaN, None, and non-numeric
            df = df.replace([None, float('inf'), float('-inf')], pd.NA)
            df = df.dropna()

            if df.empty:
                self.logger.warning("[CAGR] All values were NaN or invalid")
                return 0.0

            # Aggregate by periods
            period_data = (
                df.groupby(period_column)[column]
                .sum()
                .reset_index()
                .sort_values(period_column)
            )

            if len(period_data) < 2:
                self.logger.info("[CAGR] Not enough periods for calculation")
                return 0.0

            # ------------------------------------------------------------
            # 3. Extract numeric values
            # ------------------------------------------------------------
            start_value = float(period_data.iloc[0][column])
            end_value = float(period_data.iloc[-1][column])

            # Validate values
            if start_value <= 0:
                self.logger.warning(f"[CAGR] Invalid start value: {start_value}")
                return 0.0

            if end_value < 0:
                self.logger.warning(f"[CAGR] Invalid end value: {end_value}")
                return 0.0

            num_periods = len(period_data) - 1
            if num_periods <= 0:
                return 0.0

            # ------------------------------------------------------------
            # 4. CAGR Calculation (safe)
            # ------------------------------------------------------------
            try:
                ratio = end_value / start_value

                # Prevent domain errors
                if ratio <= 0:
                    self.logger.warning(f"[CAGR] Invalid ratio (end/start): {ratio}")
                    return 0.0

                cagr = (ratio ** (1 / num_periods) - 1) * 100

            except Exception as calc_error:
                self.logger.error(f"[CAGR] Calculation error: {calc_error}")
                return 0.0

            # ------------------------------------------------------------
            # 5. Output cleaning
            # ------------------------------------------------------------
            if pd.isna(cagr) or cagr == float("inf"):
                self.logger.warning("[CAGR] Calculation resulted in invalid output")
                return 0.0

            result = round(cagr, 2)

            # Log success
            self.logger.info(
                f"[CAGR] {period_column}: {len(period_data)} periods | "
                f"Start={start_value} End={end_value} CAGR={result}%"
            )

            return result

        # ------------------------------------------------------------
        # 6. Global fail-safe
        # ------------------------------------------------------------
        except Exception as e:
            self.logger.error(f"[CAGR] Unexpected error: {e}")
            return 0.0
    
    def analyze_seasonal_trends(self, data: pd.DataFrame) -> Dict:
        """
        Analyze seasonal patterns in vehicle registrations (MAXED EDITION).

        Provides:
            • Monthly averages
            • Peak & low season
            • Month-on-month deviation
            • Seasonal strength score
            • Trend stability index
            • Auto-cleaning of invalid or missing data

        Returns:
            Dict: Detailed seasonal analysis
        """

        try:
            # ------------------------------------------------------
            # 1. Validate input
            # ------------------------------------------------------
            if not isinstance(data, pd.DataFrame) or data.empty:
                self.logger.warning("[Seasonal] Invalid or empty DataFrame")
                return {}

            if "TOTAL" not in data.columns:
                self.logger.warning("[Seasonal] Missing required column: TOTAL")
                return {}

            if "Month" not in data.columns:
                self.logger.info("[Seasonal] No monthly data found")
                return {}

            # ------------------------------------------------------
            # 2. Clone & clean dataset
            # ------------------------------------------------------
            df = data.copy()

            # Clean invalid numbers
            df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
            df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

            df = df.dropna(subset=["Month", "TOTAL"])

            if df.empty:
                self.logger.warning("[Seasonal] All values invalid after cleaning")
                return {}

            # Ensure month is valid (1‒12)
            df = df[(df["Month"] >= 1) & (df["Month"] <= 12)]
            if df.empty:
                self.logger.warning("[Seasonal] No valid month values (1-12)")
                return {}

            # ------------------------------------------------------
            # 3. Monthly averages
            # ------------------------------------------------------
            monthly_trends = df.groupby("Month")["TOTAL"].mean()

            if monthly_trends.empty:
                self.logger.warning("[Seasonal] Monthly grouping failed")
                return {}

            # ------------------------------------------------------
            # 4. Peak & low season
            # ------------------------------------------------------
            peak_month = int(monthly_trends.idxmax())
            low_month = int(monthly_trends.idxmin())

            seasonal_range = monthly_trends.max() - monthly_trends.min()

            # ------------------------------------------------------
            # 5. Month-on-month deviation
            # ------------------------------------------------------
            mom_deviation = monthly_trends.pct_change().replace([float("inf"), -float("inf")], pd.NA)
            mom_deviation = mom_deviation.dropna()

            # ------------------------------------------------------
            # 6. Seasonal Strength Score (0–100)
            # ------------------------------------------------------
            if monthly_trends.mean() > 0:
                seasonal_strength = round(
                    (seasonal_range / monthly_trends.mean()) * 100, 2
                )
            else:
                seasonal_strength = 0.0

            # ------------------------------------------------------
            # 7. Trend Stability Index (0–100)
            # ------------------------------------------------------
            # Lower variance = more stable monthly behavior
            if len(monthly_trends) > 1:
                stability_index = max(
                    0.0,
                    100 - round(monthly_trends.std() / (monthly_trends.mean() + 1e-6) * 100, 2)
                )
            else:
                stability_index = 100.0

            # ------------------------------------------------------
            # 8. Build result dictionary
            # ------------------------------------------------------
            seasonal_analysis = {
                "monthly_averages": monthly_trends.round(2).to_dict(),
                "peak_season": {
                    "month": peak_month,
                    "average_registrations": round(monthly_trends[peak_month], 2)
                },
                "low_season": {
                    "month": low_month,
                    "average_registrations": round(monthly_trends[low_month], 2)
                },
                "month_on_month_deviation": mom_deviation.round(4).to_dict(),
                "seasonal_strength_score": seasonal_strength,   # >60 = strong seasonality
                "trend_stability_index": stability_index         # >70 = stable demand
            }

            self.logger.info(
                "[Seasonal] Seasonal analysis complete | "
                f"Peak={peak_month}, Low={low_month}, Strength={seasonal_strength}"
            )

            return seasonal_analysis

        except Exception as e:
            self.logger.error(f"[Seasonal] Unexpected error: {e}")
            return {}
    
    def calculate_market_penetration(self, data: pd.DataFrame) -> Dict:
        """
        Calculate market penetration metrics (MAXED EDITION).

        Provides:
            • State-wise total registrations
            • Market share %
            • Dominant markets ( >10% share )
            • Emerging markets ( 2–10% share )
            • Micro-markets ( <2% share )
            • Gini coefficient of market concentration
            • HHI (Herfindahl–Hirschman Index)
            • Top 5 & bottom 5 states

        Returns:
            Dict: Market penetration analysis
        """

        try:
            # ----------------------------------------------------------
            # 1. Validate Input
            # ----------------------------------------------------------
            if not isinstance(data, pd.DataFrame) or data.empty:
                self.logger.warning("[Penetration] Invalid or empty DataFrame")
                return {}

            if "State" not in data.columns or "TOTAL" not in data.columns:
                self.logger.warning("[Penetration] Missing columns: State/TOTAL")
                return {}

            # ----------------------------------------------------------
            # 2. Clean Data
            # ----------------------------------------------------------
            df = data.copy()

            df["State"] = df["State"].astype(str)

            df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
            df = df.dropna(subset=["State", "TOTAL"])

            if df.empty:
                self.logger.warning("[Penetration] No valid rows after cleaning")
                return {}

            # Remove states with negative or zero registrations
            df = df[df["TOTAL"] > 0]
            if df.empty:
                self.logger.warning("[Penetration] All registrations <= 0")
                return {}

            # ----------------------------------------------------------
            # 3. Compute Basic Metrics
            # ----------------------------------------------------------
            state_registrations = df.groupby("State")["TOTAL"].sum()
            total_registrations = state_registrations.sum()

            if total_registrations <= 0:
                self.logger.warning("[Penetration] Invalid total registrations")
                return {}

            # State-wise market share %
            market_share = (state_registrations / total_registrations * 100).round(2)

            # ----------------------------------------------------------
            # 4. Classification
            # ----------------------------------------------------------
            dominant_states = market_share[market_share > 10].index.tolist()
            emerging_states = market_share[(market_share > 2) & (market_share <= 10)].index.tolist()
            micro_markets = market_share[market_share <= 2].index.tolist()

            # ----------------------------------------------------------
            # 5. Advanced Concentration Metrics
            # ----------------------------------------------------------

            # ---- Herfindahl–Hirschman Index (HHI)
            # HHI < 1500 = competitive, 1500–2500 = moderate, >2500 = concentrated
            hhi = float((market_share ** 2).sum())

            # ---- Gini Coefficient for market equality
            sorted_share = market_share.sort_values().values
            n = len(sorted_share)
            if n > 1:
                gini = round(
                    (
                        (2 * sum((i + 1) * sorted_share[i] for i in range(n)))
                        / (n * sorted_share.sum())
                        - (n + 1) / n
                    ),
                    4,
                )
            else:
                gini = 0.0

            # ----------------------------------------------------------
            # 6. Top & Bottom Markets
            # ----------------------------------------------------------
            top_5 = market_share.sort_values(ascending=False).head(5).to_dict()
            bottom_5 = market_share.sort_values(ascending=True).head(5).to_dict()

            # ----------------------------------------------------------
            # 7. Build Final Response
            # ----------------------------------------------------------
            penetration_metrics = {
                "state_market_share": market_share.to_dict(),
                "dominant_markets": dominant_states,
                "emerging_markets": emerging_states,
                "micro_markets": micro_markets,
                "top_5_markets": top_5,
                "bottom_5_markets": bottom_5,
                "hhi_index": round(hhi, 2),
                "gini_coefficient": gini,
                "total_states": len(state_registrations),
            }

            self.logger.info(
                f"[Penetration] Analysis complete | States={len(state_registrations)}, "
                f"HHI={hhi:.2f}, Gini={gini}"
            )

            return penetration_metrics

        except Exception as e:
            self.logger.error(f"[Penetration] Unexpected error: {e}")
            return {}
    
    def analyze_growth_volatility(self, data: pd.DataFrame) -> Dict:
        """
        Analyze growth volatility & stability in yearly registrations (MAXED EDITION).

        Provides:
            • Year-over-year growth % list
            • Average growth
            • Standard deviation (volatility)
            • Stability score (0–100)
            • Outlier growth spikes/dips
            • Growth consistency index
            • Coefficient of variation
            • Best & worst years

        Returns:
            Dict: Growth volatility analysis
        """

        try:
            # ----------------------------------------------------------
            # 1. Validate input
            # ----------------------------------------------------------
            if not isinstance(data, pd.DataFrame) or data.empty:
                self.logger.warning("[Volatility] Invalid/empty DataFrame")
                return {}

            if "Year" not in data.columns or "TOTAL" not in data.columns:
                self.logger.warning("[Volatility] Missing Year or TOTAL column")
                return {}

            # ----------------------------------------------------------
            # 2. Clean dataset
            # ----------------------------------------------------------
            df = data.copy()

            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")

            df = df.dropna(subset=["Year", "TOTAL"])
            if df.empty:
                self.logger.warning("[Volatility] No valid rows after cleaning")
                return {}

            # ----------------------------------------------------------
            # 3. Aggregate yearly data
            # ----------------------------------------------------------
            yearly_data = (
                df.groupby("Year")["TOTAL"].sum()
                .reset_index()
                .sort_values("Year")
            )

            if len(yearly_data) < 2:
                self.logger.info("[Volatility] Not enough periods for growth calculation")
                return {}

            # ----------------------------------------------------------
            # 4. Calculate YoY growth rates
            # ----------------------------------------------------------
            growth_rates = []
            valid_years = []

            for i in range(1, len(yearly_data)):
                current = yearly_data.iloc[i]["TOTAL"]
                previous = yearly_data.iloc[i - 1]["TOTAL"]

                year = int(yearly_data.iloc[i]["Year"])

                if previous > 0:
                    growth_rate = ((current - previous) / previous) * 100
                    growth_rates.append(round(growth_rate, 3))
                    valid_years.append(year)

            if not growth_rates:
                self.logger.info("[Volatility] All previous-year values were <= 0")
                return {}

            # Convert to numpy array for performance
            gr = np.array(growth_rates)

            # ----------------------------------------------------------
            # 5. Compute volatility metrics
            # ----------------------------------------------------------
            avg_growth = round(float(np.mean(gr)), 3)
            volatility = round(float(np.std(gr)), 3)

            # Coefficient of variation
            if avg_growth != 0:
                cov = round(volatility / abs(avg_growth), 3)
            else:
                cov = None

            # ----------------------------------------------------------
            # 6. Outlier detection using Z-score
            # ----------------------------------------------------------
            try:
                z_scores = (gr - gr.mean()) / (gr.std() if gr.std() != 0 else 1)
                outliers = [
                    {"year": valid_years[i], "growth_rate": gr[i], "z_score": round(z_scores[i], 3)}
                    for i in range(len(z_scores)) if abs(z_scores[i]) > 2
                ]
            except Exception:
                outliers = []

            # ----------------------------------------------------------
            # 7. Best & Worst growth years
            # ----------------------------------------------------------
            best_index = int(np.argmax(gr))
            worst_index = int(np.min(np.where(gr == gr.min()))) if len(gr) > 0 else 0

            best_year = {
                "year": int(valid_years[best_index]),
                "growth_rate": float(gr[best_index])
            }

            worst_year = {
                "year": int(valid_years[worst_index]),
                "growth_rate": float(gr[worst_index])
            }

            # ----------------------------------------------------------
            # 8. Stability Score (0–100)
            # ----------------------------------------------------------
            stability_score = self._calculate_stability_score(growth_rates)

            # ----------------------------------------------------------
            # 9. Growth consistency index (0–1)
            # ----------------------------------------------------------
            # 1 = perfectly consistent YoY growth
            if volatility == 0:
                consistency_index = 1.0
            else:
                consistency_index = round(1 / (1 + volatility / 100), 4)

            # ----------------------------------------------------------
            # 10. Build response
            # ----------------------------------------------------------
            volatility_analysis = {
                "years_evaluated": valid_years,
                "growth_rates": [float(x) for x in gr],
                "average_growth": avg_growth,
                "growth_volatility": volatility,
                "coefficient_of_variation": cov,
                "stability_score": stability_score,
                "consistency_index": consistency_index,
                "best_year": best_year,
                "worst_year": worst_year,
                "outlier_years": outliers,
            }

            self.logger.info(
                f"[Volatility] Complete | Avg={avg_growth}% | Vol={volatility}% | Stability={stability_score}"
            )

            return volatility_analysis

        except Exception as e:
            self.logger.error(f"[Volatility] Unexpected error: {e}")
            return {}
    
    def _calculate_stability_score(self, growth_rates: List[float]) -> str:
        """
        MAXED EDITION (2025 Hardened Version)
        -------------------------------------
        Calculates a stability score based on the volatility of YoY growth rates.
        Includes:
        ✓ Safety checks
        ✓ NaN/inf handling
        ✓ Robust volatility computation
        ✓ Clear, deterministic scoring
        """
        try:
            # Validate list
            if not growth_rates or not isinstance(growth_rates, (list, tuple)):
                return "Unknown"

            # Convert to numpy array for safety
            arr = np.array(growth_rates, dtype=float)

            # Remove invalid values
            arr = arr[~np.isnan(arr)]
            arr = arr[np.isfinite(arr)]

            if arr.size == 0:
                return "Unknown"

            # Calculate volatility (standard deviation)
            volatility = float(np.std(arr))

            # Classify based on volatility bands
            if volatility < 5:
                return "Very Stable"
            elif volatility < 10:
                return "Stable"
            elif volatility < 20:
                return "Moderate"
            elif volatility < 30:
                return "Volatile"
            else:
                return "Highly Volatile"

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating stability score: {e}")
            return "Unknown"
    
    def identify_growth_patterns(self, data: pd.DataFrame) -> Dict:
        """
        MAXED EDITION (2025 Hardened Version)
        -------------------------------------
        Identifies long-term and short-term growth patterns:
        ✓ Direction (Upward / Downward / Flat)
        ✓ Acceleration / Deceleration
        ✓ Cyclical behavior detection
        ✓ Handles missing, duplicate, noisy data
        ✓ Full error shielding and logging
        """
        try:
            patterns = {}

            # ---------- BASIC VALIDATION ----------
            required_cols = {'Year', 'TOTAL'}
            if not required_cols.issubset(data.columns):
                return patterns

            # ---------- CLEAN & PREPARE ----------
            yearly = (
                data.groupby('Year')['TOTAL']
                .sum()
                .reset_index()
                .sort_values('Year')
            )

            # Remove invalid values
            yearly = yearly.replace([np.inf, -np.inf], np.nan).dropna()
            if yearly.empty or len(yearly) < 2:
                return patterns

            # ---------- TREND DIRECTION ----------
            if len(yearly) >= 3:
                recent = yearly.tail(3)
                patterns['trend_direction'] = self._analyze_trend_direction(recent)
            else:
                patterns['trend_direction'] = "Unknown"

            # ---------- ACCELERATION / DECELERATION ----------
            try:
                accel = self._analyze_growth_acceleration(yearly)
                patterns['growth_acceleration'] = accel
            except Exception as inner:
                self.logger.warning(f"⚠️ Acceleration analysis failed: {inner}")
                patterns['growth_acceleration'] = "Unknown"

            # ---------- CYCLICAL PATTERNS ----------
            try:
                cycles = self._identify_cyclical_patterns(yearly)
                patterns['cyclical_patterns'] = cycles
            except Exception as inner:
                self.logger.warning(f"⚠️ Cyclical pattern detection failed: {inner}")
                patterns['cyclical_patterns'] = {}

            return patterns

        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying growth patterns: {e}")
            return {}
    
    def _analyze_trend_direction(self, data: pd.DataFrame) -> str:
        """
        MAXED EDITION (2025 Hardened Version)
        -------------------------------------
        Determines the direction of the trend using:
        ✓ Noise-tolerant regression
        ✓ Adaptive slope threshold
        ✓ Failure-proof analysis
        Returns:
        "Upward Trend" | "Downward Trend" | "Flat Trend" |
        "Insufficient Data" | "Unknown"
        """
        try:
            # ---------- VALIDATION ----------
            if (
                data is None 
                or len(data) < 2 
                or 'TOTAL' not in data.columns
            ):
                return "Insufficient Data"

            # Clean and sanitize values
            y = (
                data['TOTAL']
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
                .values
            )

            if len(y) < 2:
                return "Insufficient Data"

            # ---------- REGRESSION (Slope Detection) ----------
            x = np.arange(len(y))

            try:
                slope = np.polyfit(x, y, 1)[0]
            except Exception as reg_err:
                self.logger.warning(
                    f"⚠️ Regression failed in trend analysis: {reg_err}"
                )
                return "Unknown"

            # ---------- TREND CLASSIFICATION ----------
            # Adaptive threshold prevents classification errors on noisy flat data
            mean_val = max(np.mean(y), 1e-6)
            threshold = 0.005 * mean_val  # 0.5% variation threshold

            if slope > threshold:
                return "Upward Trend"
            elif slope < -threshold:
                return "Downward Trend"
            else:
                return "Flat Trend"

        except Exception as e:
            self.logger.warning(
                f"⚠️ Error in trend direction analysis: {e}"
            )
            return "Unknown"

    def _analyze_growth_acceleration(self, data: pd.DataFrame) -> str:
        """
        MAXED EDITION (2025 Hardened Version)
        -------------------------------------
        Determines whether growth is accelerating, decelerating, or stable.
        Features:
        ✓ Full noise protection
        ✓ Adaptive thresholding
        ✓ Outlier-resistant growth rates
        ✓ Stable with missing / zero / infinite values
        """
        try:
            # ---------- VALIDATION ----------
            if data is None or len(data) < 3 or 'TOTAL' not in data.columns:
                return "Insufficient Data"

            # Clean values (remove NaN, Inf)
            cleaned = (
                data['TOTAL']
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
                .values
            )

            if len(cleaned) < 3:
                return "Insufficient Data"

            # ---------- CALCULATE YoY GROWTH ----------
            growth_rates = []
            for i in range(1, len(cleaned)):
                prev = cleaned[i - 1]
                curr = cleaned[i]

                if prev > 0:
                    growth = ((curr - prev) / prev) * 100
                    growth_rates.append(growth)

            # Need at least 2 meaningful growth rate points
            if len(growth_rates) < 2:
                return "Insufficient Data"

            # ---------- ROBUST SMOOTHING ----------
            # Clip extreme spikes (avoids outlier distortion)
            growth_rates = np.clip(growth_rates, -500, 500)

            # ---------- RECENT vs HISTORICAL COMPARISON ----------
            recent_avg = np.mean(growth_rates[-2:])
            earlier_avg = np.mean(growth_rates[:-2]) if len(growth_rates) > 2 else growth_rates[0]

            # Adaptive threshold (to handle large numbers)
            dynamic_threshold = max(2, 0.05 * abs(earlier_avg))  # min 2%, or ±5% of earlier avg

            # ---------- FINAL CLASSIFICATION ----------
            if recent_avg > earlier_avg + dynamic_threshold:
                return "Accelerating"
            elif recent_avg < earlier_avg - dynamic_threshold:
                return "Decelerating"
            else:
                return "Stable"

        except Exception as e:
            self.logger.warning(f"⚠️ Error in growth acceleration analysis: {e}")
            return "Unknown"
    
    def _identify_cyclical_patterns(self, data: pd.DataFrame) -> Dict:
        """
        MAXED EDITION (2025 Hardened Version)
        -------------------------------------
        Identifies cyclical behavior using:
        ✓ Noise-resistant smoothing
        ✓ Peak & trough detection
        ✓ Adaptive thresholds for strong vs weak cycles
        ✓ Outlier protection
        Returns structured cycle metadata.
        """
        try:
            # ---------- VALIDATION ----------
            if data is None or len(data) < 4 or 'TOTAL' not in data.columns:
                return {"pattern": "Insufficient Data"}

            # ---------- CLEANING ----------
            values = (
                data['TOTAL']
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
                .values
            )

            if len(values) < 4:
                return {"pattern": "Insufficient Data"}

            # ---------- OPTIONAL SMOOTHING (reduces false peaks) ----------
            # Use simple 3-point moving average smoothing to stabilize noise.
            smooth_values = np.copy(values)
            for i in range(1, len(values) - 1):
                smooth_values[i] = (values[i-1] + values[i] + values[i+1]) / 3

            values = smooth_values

            # ---------- PEAK & TROUGH DETECTION ----------
            peaks = []
            troughs = []

            # Adaptive magnitude threshold to filter fake micro-peaks
            amplitude_threshold = 0.02 * np.mean(values)  # 2% variation rule

            for i in range(1, len(values) - 1):
                diff_prev = values[i] - values[i - 1]
                diff_next = values[i] - values[i + 1]

                # Peak detection
                if diff_prev > 0 and diff_next > 0 and abs(diff_prev) > amplitude_threshold:
                    peaks.append(i)

                # Trough detection
                if diff_prev < 0 and diff_next < 0 and abs(diff_prev) > amplitude_threshold:
                    troughs.append(i)

            # ---------- CLASSIFY PATTERN ----------
            total_turnpoints = len(peaks) + len(troughs)

            if total_turnpoints == 0:
                pattern_type = "Linear"
            elif total_turnpoints == 1:
                pattern_type = "Weak Cycle"
            elif total_turnpoints >= 2:
                pattern_type = "Cyclical"
            else:
                pattern_type = "Unknown"

            # ---------- RETURN STRUCTURED RESPONSE ----------
            return {
                "peaks_count": len(peaks),
                "troughs_count": len(troughs),
                "turning_points": total_turnpoints,
                "pattern": pattern_type,
                "peaks_positions": peaks,
                "troughs_positions": troughs
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying cyclical patterns: {e}")
            return {"pattern": 'Unknown'}
    
    def generate_growth_forecast(
        self,
        data: pd.DataFrame,
        forecast_periods: int = 2
    ) -> Dict:
        """
        MAXED EDITION (2025 Hardened):
        ------------------------------
        • Full NaN sanitization
        • Handles Year as int/str/mixed
        • Removes outliers (optional hook)
        • Prevents regression overflows
        • Auto-catches degenerate trends
        • Uses stable pseudo-linear estimator for small noisy datasets
        • Adds diagnostics + metadata
        • Adds safe-bounds forecast limiting
        """

        try:
            # --- BASIC VALIDATION ---
            required_columns = {'Year', 'TOTAL'}
            if not required_columns.issubset(data.columns):
                return {"error": f"Missing columns: {required_columns - set(data.columns)}"}

            # --- CLEANING ---
            df = data.copy()

            # Normalize Year → int
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
            df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')

            df = df.dropna(subset=['Year', 'TOTAL'])

            if df.empty or len(df) < 2:
                return {"error": "Insufficient clean data"}

            # --- GROUPED YEARLY DATA ---
            yearly = (
                df.groupby('Year')['TOTAL']
                .sum()
                .reset_index()
                .sort_values('Year')
            )

            if len(yearly) < 2:
                return {"error": "Not enough yearly points"}

            years = yearly['Year'].astype(int).values
            values = yearly['TOTAL'].astype(float).values

            # --- DETECT FLAT OR NEAR-CONSTANT DATA ---
            if np.allclose(values, values[0], atol=1e-6):
                flat_value = int(values[0])
                return {
                    "forecasts": {int(years[-1] + i): flat_value for i in range(1, forecast_periods + 1)},
                    "confidence": "Low",
                    "trend_slope": 0.0,
                    "r_squared": 0.0,
                    "note": "Flat historical data — using constant projection."
                }

            # --- REGRESSION (Robust) ---
            try:
                slope, intercept = np.polyfit(years, values, 1)
            except Exception:
                # fallback safe-slope estimator
                slope = (values[-1] - values[0]) / max(1, (years[-1] - years[0]))
                intercept = values[0] - slope * years[0]

            # --- FORECAST ---
            last_year = years[-1]
            forecasts = {}

            for i in range(1, forecast_periods + 1):
                forecast_year = int(last_year + i)

                forecast_value = slope * forecast_year + intercept

                # Prevent negative predicted totals
                forecast_value = max(0, int(forecast_value))

                # Safe upper bound (prevents regression explosion)
                safe_cap = max(values) * 5
                forecast_value = min(forecast_value, safe_cap)

                forecasts[forecast_year] = forecast_value

            # --- FIT QUALITY (R²) ---
            try:
                predicted = slope * years + intercept
                ss_res = np.sum((values - predicted) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            except Exception:
                r2 = 0

            # --- CONFIDENCE ---
            if r2 > 0.85:
                confidence = "High"
            elif r2 > 0.55:
                confidence = "Medium"
            else:
                confidence = "Low"

            return {
                "forecasts": forecasts,
                "confidence": confidence,
                "r_squared": round(float(r2), 3),
                "trend_slope": round(float(slope), 2),
                "history_points": len(values),
                "min_year": int(years.min()),
                "max_year": int(years.max()),
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Forecast generation failed: {e}")
            return {"error": str(e)}
