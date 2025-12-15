import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
from datetime import datetime
from nsepython import nse_eq, nsefetch
import time
from dash.dependencies import ALL
import plotly.graph_objects as go
import json
import logging
import os
from cachetools import TTLCache, cached
import pytz
from urllib.parse import quote

# ===================================================================
# CONFIGURATION CONSTANTS
# ===================================================================
TRADING_DAYS_BUFFER = 10          # Extra days to fetch for weekends/holidays
CRORE_DIVISOR = 10_000_000        # 1 Crore = 10 million
DEFAULT_BATCH_SIZE = 50           # Stocks per batch
DEFAULT_WORKERS = 5               # Concurrent threads
CACHE_TTL_SECONDS = 300           # 5 minutes
CACHE_MAX_SIZE = 256              # Max cached items

# Volume significance thresholds
VOL_ELEVATED_THRESHOLD = 1.20     # 20% above average = elevated
VOL_REDUCED_THRESHOLD = 0.80      # 20% below average = reduced
VOL_AVERAGE_DAYS = 10             # Days to use for volume average calculation

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Timezone
IST = pytz.timezone('Asia/Kolkata')

# Supported Indices
SUPPORTED_INDICES = {
    "NIFTY50": {"name": "Nifty 50", "emoji": "📊", "count": 50},
    "NIFTY100": {"name": "Nifty 100", "emoji": "📈", "count": 100},
}
DEFAULT_INDEX = "NIFTY100"

# Intraday volume distribution (cumulative expected % by time)
# Based on typical Indian market volume patterns
INTRADAY_VOL_DISTRIBUTION = {
    (9, 15): 0.00, (9, 30): 0.05, (9, 45): 0.10, (10, 0): 0.15,
    (10, 15): 0.19, (10, 30): 0.23, (10, 45): 0.27, (11, 0): 0.31,
    (11, 15): 0.35, (11, 30): 0.38, (11, 45): 0.41, (12, 0): 0.44,
    (12, 15): 0.47, (12, 30): 0.50, (12, 45): 0.53, (13, 0): 0.56,
    (13, 15): 0.59, (13, 30): 0.62, (13, 45): 0.65, (14, 0): 0.69,
    (14, 15): 0.73, (14, 30): 0.78, (14, 45): 0.84, (15, 0): 0.91,
    (15, 15): 0.96, (15, 30): 1.00,
}

# NSE Index API URLs - Official NSE endpoints for index constituents
NSE_INDEX_API_URL = "https://www.nseindia.com/api/equity-stockIndices?index={}"
NSE_INDEX_NAMES = {
    "NIFTY50": "NIFTY 50",
    "NIFTY100": "NIFTY 100",
}

# Yahoo Finance tickers for index OHLC data
INDEX_YF_TICKERS = {
    "NIFTY50": "^NSEI",      # NIFTY 50 on Yahoo Finance
    "NIFTY100": "^CNX100",   # NIFTY 100 on Yahoo Finance
}

# Candlestick chart configuration
INDEX_CANDLE_DAYS = 60      # 60 days = ~3 months trading, ideal for swing trading analysis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================================================================
# TTL CACHES (Auto-expire after 5 minutes)
# ===================================================================
nse_data_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
fundamentals_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
historical_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
volume_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)

# Index constituents cache (long TTL - 1 hour since index composition rarely changes)
INDEX_CACHE_TTL = 3600  # 1 hour
index_constituents_cache = TTLCache(maxsize=10, ttl=INDEX_CACHE_TTL)

# Threading lock to prevent concurrent duplicate API calls for same index
import threading
_industry_load_lock = threading.Lock()

# -------------------------------------------------------------------
# SAFETY WRAPPER
# -------------------------------------------------------------------
def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_float(v):
    """Try to convert v to float, return None if not possible (handles 'NA', None, empty)."""
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "" or s.upper() == "NA":
            return None
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return None


def retry_with_backoff(func, symbol: str, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function with exponential backoff for rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func(symbol)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Rate limited on {symbol}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"  Max retries exceeded for {symbol}: {e}")
                    return None
            else:
                raise
    return None


# -------------------------------------------------------------------
# 1) DYNAMIC INDEX LOADING - USING NSE OFFICIAL API
# -------------------------------------------------------------------
def get_index_constituents_from_nse(index_name: str) -> list:
    """
    Fetch index constituents directly from NSE Official API.
    Returns up-to-date symbols without any manual remapping needed.
    
    This is the authoritative source - NSE always has current symbols
    reflecting all corporate actions (mergers, delistings, renames).
    """
    try:
        # Get the NSE index name (with proper formatting)
        nse_index_name = NSE_INDEX_NAMES.get(index_name, "NIFTY 100")
        
        # URL encode the index name for the API
        encoded_index = quote(nse_index_name)
        url = NSE_INDEX_API_URL.format(encoded_index)
        
        logger.info(f"📡 Fetching {index_name} constituents from NSE API...")
        response = nsefetch(url)
        
        if response and isinstance(response, dict) and 'data' in response:
            symbols = []
            for item in response['data']:
                symbol = item.get('symbol')
                # Skip the index row itself (e.g., "NIFTY 50" or "NIFTY 100")
                if symbol and symbol not in ['NIFTY 50', 'NIFTY 100', 'NIFTY50', 'NIFTY100']:
                    symbols.append(symbol)
            
            if symbols:
                logger.info(f"✓ Loaded {len(symbols)} symbols from NSE Official API for {index_name}")
                return symbols
            else:
                logger.warning(f"NSE API returned empty symbols list for {index_name}")
        else:
            logger.warning(f"Invalid response from NSE API: {type(response)}")
            
    except Exception as e:
        logger.warning(f"NSE Official API failed for {index_name}: {e}")
    
    return []


def get_index_constituents(index_name: str) -> list:
    """
    Get list of symbols for an index.
    
    Data source: NSE Official API (always up-to-date, authoritative)
    
    Uses caching - only fetches once per hour.
    This ensures we always have the current index composition,
    automatically handling delistings, additions, and symbol changes.
    """
    cache_key = f"symbols_{index_name}"
    
    # Check cache first
    if cache_key in index_constituents_cache:
        logger.info(f"✓ Using cached {index_name} constituents")
        return index_constituents_cache[cache_key]
    
    # Fetch from NSE Official API (authoritative, always current)
    symbols = get_index_constituents_from_nse(index_name)
    
    if symbols:
        # Successfully got symbols from NSE API
        index_constituents_cache[cache_key] = symbols
        logger.info(f"✓ Loaded {len(symbols)} symbols from NSE API for {index_name}")
        return symbols
    
    # If API fails, return empty list (no CSV fallback)
    logger.error(f"❌ Failed to load {index_name} constituents from NSE API. No data available.")
    return []


def get_symbol_industry(symbol: str) -> dict:
    """Get industry hierarchy for a single symbol from NSE API. Uses cache.
    
    Returns dict with:
        - macro: NSE macro sector (e.g., "Financial Services", "Energy")
        - sector: NSE sector (e.g., "Banks", "Oil Gas & Consumable Fuels")
        - industry: NSE industry (e.g., "Private Sector Bank")
    """
    try:
        data = get_nse_data(symbol)
        if data:
            industry_info = data.get("industryInfo", {})
            return {
                "macro": industry_info.get("macro") or "Others",
                "sector": industry_info.get("sector") or "N/A",
                "industry": industry_info.get("industry") or industry_info.get("sector") or "N/A",
            }
    except Exception as e:
        logger.warning(f"Could not get industry for {symbol}: {e}")
    return {"macro": "Others", "sector": "N/A", "industry": "N/A"}


def load_index_with_industries(index_name: str) -> dict:
    """
    Load symbols with industries for a given index.
    
    OPTIMIZATION: Since NIFTY 50 is a subset of NIFTY 100, we:
    1. Load industries for NIFTY 100 first (100 API calls)
    2. For NIFTY 50, reuse the NIFTY 100 industry mapping (0 API calls)
    
    Uses threading lock to prevent concurrent duplicate API calls.
    This eliminates 50 redundant API calls per startup.
    
    Returns: {symbol: industry} mapping
    """
    cache_key = f"index_industries_{index_name}"
    
    # Check cache first (before acquiring lock for efficiency)
    if cache_key in index_constituents_cache:
        logger.info(f"✓ Using cached {index_name} industries mapping")
        return index_constituents_cache[cache_key]
    
    # Acquire lock to prevent concurrent duplicate fetches
    with _industry_load_lock:
        # Double-check cache after acquiring lock (another thread may have populated it)
        if cache_key in index_constituents_cache:
            logger.info(f"✓ Using cached {index_name} industries mapping (after lock)")
            return index_constituents_cache[cache_key]
    
        # Get symbols for the index from NSE API
        symbols = get_index_constituents(index_name)
        if not symbols:
            return {}
        
        symbol_industry_map = {}
        
        # OPTIMIZATION: For NIFTY50, try to reuse NIFTY100's industry mapping
        if index_name == "NIFTY50":
            nifty100_cache_key = "index_industries_NIFTY100"
            
            # If NIFTY100 cache doesn't exist, load it first (this ensures optimization always works)
            if nifty100_cache_key not in index_constituents_cache:
                logger.info(f"⚡ OPTIMIZATION: Loading NIFTY100 first to reuse for NIFTY50...")
                # Release lock temporarily to avoid deadlock, then reacquire
                # Actually we can just call it since it will return from cache check
                pass
            
            # Check again after potential load
            if nifty100_cache_key not in index_constituents_cache:
                # Load NIFTY100 within the same lock context (won't deadlock since we check cache first)
                nifty100_symbols = get_index_constituents("NIFTY100")
                if nifty100_symbols:
                    logger.info(f"📡 Fetching industries for {len(nifty100_symbols)} NIFTY100 symbols...")
                    nifty100_map = {}
                    for i, symbol in enumerate(nifty100_symbols):
                        nifty100_map[symbol] = get_symbol_industry(symbol)
                        if (i + 1) % 25 == 0:
                            logger.info(f"  Progress: {i + 1}/{len(nifty100_symbols)} industries fetched")
                    logger.info(f"✓ Loaded industries for {len(nifty100_map)} NIFTY100 stocks")
                    index_constituents_cache["index_industries_NIFTY100"] = nifty100_map
            
            # Now NIFTY100 cache should exist
            if nifty100_cache_key in index_constituents_cache:
                nifty100_industries = index_constituents_cache[nifty100_cache_key]
                logger.info(f"⚡ OPTIMIZATION: Reusing NIFTY100 industries for NIFTY50 (saving 50 API calls)")
                
                # Map NIFTY50 symbols using NIFTY100's cached industries
                missing_symbols = []
                for symbol in symbols:
                    if symbol in nifty100_industries:
                        symbol_industry_map[symbol] = nifty100_industries[symbol]
                    else:
                        missing_symbols.append(symbol)
                
                # Fetch only missing symbols (rare edge case)
                if missing_symbols:
                    logger.info(f"  Fetching {len(missing_symbols)} missing symbols...")
                    for symbol in missing_symbols:
                        industry = get_symbol_industry(symbol)
                        symbol_industry_map[symbol] = industry
                
                logger.info(f"✓ Loaded {len(symbol_industry_map)} industries for NIFTY50 (reused from NIFTY100)")
                index_constituents_cache[cache_key] = symbol_industry_map
                return symbol_industry_map
        
        # Standard path: Fetch industries from NSE API for all symbols
        logger.info(f"📡 Fetching industries for {len(symbols)} symbols from NSE API...")
        for i, symbol in enumerate(symbols):
            industry = get_symbol_industry(symbol)
            symbol_industry_map[symbol] = industry
            
            # Progress log every 25 symbols
            if (i + 1) % 25 == 0:
                logger.info(f"  Progress: {i + 1}/{len(symbols)} industries fetched")
        
        logger.info(f"✓ Loaded industries for {len(symbol_industry_map)} stocks from NSE API")
        
        # Cache the result
        index_constituents_cache[cache_key] = symbol_industry_map
        return symbol_industry_map


# Legacy function for backward compatibility
def load_symbols_with_industries():
    """Load symbols and industries - defaults to NIFTY100."""
    return load_index_with_industries(DEFAULT_INDEX)


def preload_all_indices():
    """
    Preload symbols and industries for ALL supported indices.
    
    OPTIMIZATION: Load NIFTY100 first so NIFTY50 can reuse its industry mapping.
    This reduces total API calls from 150 to just 100.
    
    Uses threading to run in background without blocking server startup.
    """
    import threading
    
    def _preload():
        logger.info("="*60)
        logger.info("🚀 PRELOADING ALL INDICES ON STARTUP...")
        logger.info("="*60)
        
        # OPTIMIZED ORDER: Load NIFTY100 first so NIFTY50 can reuse industries
        preload_order = ["NIFTY100", "NIFTY50"]
        
        for index_name in preload_order:
            if index_name not in SUPPORTED_INDICES:
                continue
            logger.info(f"\n📡 Preloading {index_name}...")
            try:
                result = load_index_with_industries(index_name)
                logger.info(f"✓ {index_name}: {len(result)} stocks with industries loaded")
            except Exception as e:
                logger.error(f"❌ Failed to preload {index_name}: {e}")
        
        logger.info("="*60)
        logger.info("✅ PRELOAD COMPLETE - All indices ready!")
        logger.info("="*60)
    
    # Run in background thread to not block server startup
    thread = threading.Thread(target=_preload, daemon=True)
    thread.start()
    return thread


# Note: Using dcc.Store for state management (thread-safe)


# -------------------------------------------------------------------
# 2) CONSOLIDATED NSE DATA FETCH
# -------------------------------------------------------------------
@cached(cache=nse_data_cache)
def get_nse_data(symbol: str):
    """Fetch full NSE data once and cache with TTL."""
    try:
        return nse_eq(symbol)
    except Exception as e:
        logger.warning(f"NSE Data Fetch Error for {symbol}: {e}")
        return None


# -------------------------------------------------------------------
# 3) FUNDAMENTALS for single stock
# -------------------------------------------------------------------
@cached(cache=fundamentals_cache)
def get_fundamentals(symbol: str):
    try:
        data = get_nse_data(symbol)
        if not data:
            return None

        meta = safe_dict(data.get("metadata"))
        sec_info = safe_dict(data.get("securityInfo"))
        price_info = safe_dict(data.get("priceInfo"))
        industry = safe_dict(data.get("industryInfo"))

        pe = safe_float(meta.get("pdSymbolPe") or meta.get("pdSectorPe"))
        last_price = safe_float(price_info.get("lastPrice") or price_info.get("close"))

        eps = None
        try:
            if pe is not None and last_price is not None and pe != 0:
                eps = last_price / pe
        except Exception:
            eps = None

        mcap = None
        issued = sec_info.get("issuedSize") or sec_info.get("issuedShares") or sec_info.get("issuedCapital")
        issued_f = safe_float(issued)
        try:
            if issued_f is not None and last_price is not None:
                mcap = issued_f * last_price
        except Exception:
            mcap = None

        sector = industry.get("industry") or industry.get("sector")
        week = safe_dict(price_info.get("weekHighLow"))

        return {
            "P/E": round(pe, 2) if pe is not None else None,
            "EPS": round(eps, 2) if eps is not None else None,
            "Market Cap": round(mcap, 2) if mcap is not None else None,
            "Sector": sector or "N/A",
            "Macro": industry.get("macro") or "Others",  # For sector rotation tracker
            "52W High": week.get("max"),
            "52W Low": week.get("min"),
            "priceInfo": price_info,
            "lastPrice": last_price,
        }

    except Exception as e:
        print("Fundamental Fetch Error:", e)
        return None


# -------------------------------------------------------------------
# 4) HISTORICAL PRICE COMPARISON
# -------------------------------------------------------------------
def get_historical_comparison(symbol: str, days: int):
    """Get price comparison for N trading days ago (correctly indexed)."""
    # Create cache key
    cache_key = f"{symbol}_{days}"
    if cache_key in historical_cache:
        return historical_cache[cache_key]
    
    try:
        tk = yf.Ticker(symbol + ".NS")
        
        # Fetch historical data - get extra days to account for weekends/holidays
        hist = tk.history(period=f"{days + TRADING_DAYS_BUFFER}d", interval="1d")
        
        if hist.empty or len(hist) < 2:
            return None, None, None
        
        # Get current price (most recent) - index -1 is today
        current_price = float(hist['Close'].iloc[-1])
        
        # Get price from N trading days ago
        # -1 is today, -2 is yesterday, so -days is N days ago
        # (Previously used -(days+1) which was off-by-one)
        target_index = -days
        if len(hist) >= abs(target_index):
            old_price = float(hist['Close'].iloc[target_index])
        else:
            # Use oldest available if not enough data
            old_price = float(hist['Close'].iloc[0])
        
        # Calculate change
        price_change = current_price - old_price
        price_change_pct = (price_change / old_price * 100) if old_price != 0 else None
        
        result = (old_price, price_change, price_change_pct)
        historical_cache[cache_key] = result
        return result
        
    except Exception as e:
        logger.warning(f"Historical data error for {symbol}: {e}")
        return None, None, None


# -------------------------------------------------------------------
# 5) VOLUME STATS
# -------------------------------------------------------------------
def get_volume_stats(symbol: str):
    """Return volume metrics with retry backoff for rate limits and TTL caching."""
    # Check cache first
    if symbol in volume_cache:
        return volume_cache[symbol]
    
    def _fetch_volume(sym):
        tk = yf.Ticker(sym + ".NS")
        
        avg_vol = None
        avg_turnover = None      # Average daily turnover (for consistent comparison)
        todays_turnover = None   # Today's turnover from Yahoo Finance
        weekly_turnover = []     # Store last 7 days of turnover data
        
        try:
            hist = tk.history(period=f"{VOL_AVERAGE_DAYS + 5}d", interval="1d")
            if "Volume" in hist.columns and not hist.empty:
                # Use last VOL_AVERAGE_DAYS for average calculation
                vol_data = hist.tail(VOL_AVERAGE_DAYS) if len(hist) >= VOL_AVERAGE_DAYS else hist
                avg_vol = float(vol_data["Volume"].mean())
                
                # Calculate average turnover (Close × Volume) for consistency
                if "Close" in vol_data.columns:
                    turnovers = vol_data["Close"] * vol_data["Volume"]
                    avg_turnover = float(turnovers.mean())
                
                # Extract last 7 days of turnover (Close × Volume)
                if "Close" in hist.columns and len(hist) >= 1:
                    # Get today's turnover from the most recent data point
                    latest_row = hist.iloc[-1]
                    todays_turnover = float(latest_row["Close"]) * float(latest_row["Volume"])
                    
                    # Get last 7 days for the weekly chart
                    if len(hist) >= 7:
                        last_7_days = hist.tail(7)
                        for idx, row in last_7_days.iterrows():
                            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
                            turnover = float(row["Close"]) * float(row["Volume"])
                            weekly_turnover.append({"date": date_str, "turnover": turnover})
        except Exception as e:
            logger.warning(f"  Avg volume fetch error for {sym}: {e}")

        todays_vol = None
        try:
            intraday = tk.history(period="1d", interval="1m")
            if "Volume" in intraday.columns and not intraday["Volume"].empty:
                todays_vol = float(intraday["Volume"].sum())
        except Exception:
            try:
                daily = tk.history(period="1d", interval="1d")
                if "Volume" in daily.columns and not daily["Volume"].empty:
                    todays_vol = float(daily["Volume"].iloc[-1])
            except Exception:
                todays_vol = None

        vol_change_pct = None
        try:
            if avg_vol and todays_vol is not None and avg_vol != 0:
                vol_change_pct = (todays_vol - avg_vol) / avg_vol * 100.0
        except Exception:
            pass

        return {
            "avg_volume": avg_vol,
            "todays_volume": todays_vol,
            "volume_change_pct": vol_change_pct,
            "weekly_turnover": weekly_turnover,
            # NEW: Consistent turnover data from Yahoo Finance
            "todays_turnover": todays_turnover,    # For TURNOVER TODAY display
            "avg_turnover": avg_turnover,           # For comparison (30-day avg)
        }
    
    result = retry_with_backoff(_fetch_volume, symbol, max_retries=3, base_delay=0.5) or {
        "avg_volume": None,
        "todays_volume": None,
        "volume_change_pct": None,
        "weekly_turnover": [],
        "todays_turnover": None,
        "avg_turnover": None,
    }
    # Store in cache for TTL expiration
    volume_cache[symbol] = result
    return result


# -------------------------------------------------------------------
# 6) AGGREGATE VOLUME INDICATOR (Market-Wide)
# -------------------------------------------------------------------
from datetime import time as dt_time

def get_expected_volume_factor(current_time):
    """
    Get the expected cumulative volume factor based on time of day.
    Uses U-shaped distribution (high at open/close, low at midday).
    Returns a value between 0.0 and 1.0 representing expected % of daily volume.
    """
    if current_time is None:
        return 1.0
    
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # Find the nearest time slot in our distribution
    best_factor = 1.0
    for (hour, minute), factor in INTRADAY_VOL_DISTRIBUTION.items():
        if (hour, minute) <= (current_hour, current_minute):
            best_factor = factor
    
    return best_factor if best_factor > 0 else 0.01  # Avoid division by zero

def calculate_aggregate_volume_indicator(stocks_data):
    """
    Calculate aggregate volume % change indicator for all stocks.
    
    FIXED APPROACH:
    - Uses pure volume ratio for main comparison (avoids price timing issues)
    - Turnover is calculated separately for display purposes only
    - Uses significance thresholds for volume breadth (20% above/below)
    - Uses non-linear intraday volume distribution for projections
    
    Returns:
        dict with volume metrics, alert level, and market breadth
    """
    if not stocks_data:
        return None
    
    # Initialize accumulators
    total_volume_today = 0            # Raw volume today (for ratio)
    total_volume_avg = 0              # Raw average volume (for ratio)
    total_turnover_today = 0          # Price × Today's Volume (for display)
    total_turnover_avg = 0            # Price × Average Volume (for display)
    
    # Directional flow tracking
    up_turnover = 0                   # Turnover from stocks going up
    down_turnover = 0                 # Turnover from stocks going down
    
    # Market breadth counters with significance thresholds
    stocks_elevated_vol = 0           # Stocks with volume > 1.2× average
    stocks_normal_vol = 0             # Stocks with volume 0.8× to 1.2× average
    stocks_reduced_vol = 0            # Stocks with volume < 0.8× average
    stocks_up = 0
    stocks_down = 0
    stocks_neutral = 0
    
    # For weighted calculations
    total_market_cap = 0
    weighted_vol_change = 0
    
    valid_stocks = 0
    individual_vol_changes = []       # For statistical analysis
    
    for stock in stocks_data:
        # Get required data
        current_price = safe_float(stock.get("TODAY_CURRENT_PRICE"))
        today_volume = safe_float(stock.get("TODAY_VOLUME"))
        avg_volume = safe_float(stock.get("TODAY_VOLUME_AVERAGE"))
        price_change_pct = safe_float(stock.get("TODAY_CURRENT_PRICE_CHANGE_PCT"))
        market_cap = safe_float(stock.get("MARKET_CAP_CR"))
        vol_change_pct = safe_float(stock.get("VOL_CHANGE_PCT"))
        
        # NEW: Get consistent turnover from Yahoo Finance (same source as weekly chart)
        todays_turnover_yf = safe_float(stock.get("TODAYS_TURNOVER"))
        avg_turnover_yf = safe_float(stock.get("AVG_TURNOVER"))
        
        # Skip if essential data is missing
        if current_price is None or today_volume is None or avg_volume is None:
            continue
        if current_price <= 0 or avg_volume <= 0:
            continue
        
        valid_stocks += 1
        
        # Accumulate raw volumes for ratio calculation (FIXED: pure volume comparison)
        total_volume_today += today_volume
        total_volume_avg += avg_volume
        
        # FIXED: Use Yahoo Finance turnover for consistency with weekly chart
        # This ensures TURNOVER TODAY matches the weekly chart values exactly
        if todays_turnover_yf is not None:
            total_turnover_today += todays_turnover_yf
        else:
            # Fallback to calculated if Yahoo data missing
            total_turnover_today += current_price * today_volume
            
        if avg_turnover_yf is not None:
            total_turnover_avg += avg_turnover_yf
        else:
            # Fallback to calculated if Yahoo data missing
            total_turnover_avg += current_price * avg_volume
        
        # Track individual volume changes for stats
        if vol_change_pct is not None:
            individual_vol_changes.append(vol_change_pct)
        
        # Market breadth with significance thresholds (FIXED: 20% thresholds)
        volume_ratio_stock = today_volume / avg_volume
        if volume_ratio_stock >= VOL_ELEVATED_THRESHOLD:
            stocks_elevated_vol += 1
        elif volume_ratio_stock <= VOL_REDUCED_THRESHOLD:
            stocks_reduced_vol += 1
        else:
            stocks_normal_vol += 1
        
        # Market cap weighted volume change
        if market_cap is not None and vol_change_pct is not None:
            total_market_cap += market_cap
            weighted_vol_change += market_cap * vol_change_pct
        
        # Directional flow (up vs down stocks)
        # Use Yahoo Finance turnover for consistency
        stock_turnover = todays_turnover_yf if todays_turnover_yf is not None else (current_price * today_volume)
        
        if price_change_pct is not None:
            if price_change_pct > 0:
                up_turnover += stock_turnover
                stocks_up += 1
            elif price_change_pct < 0:
                down_turnover += stock_turnover
                stocks_down += 1
            else:
                stocks_neutral += 1
        else:
            stocks_neutral += 1
    
    # Avoid division by zero
    if total_volume_avg == 0 or valid_stocks == 0:
        return None
    
    # ========== CORE CALCULATIONS ==========
    
    # 1. Volume Ratio using RAW VOLUME (FIXED: no price mixing)
    volume_ratio = total_volume_today / total_volume_avg
    volume_pct_change = (volume_ratio - 1) * 100
    
    # 2. Market Cap Weighted Volume Change
    mcap_weighted_vol_change = None
    if total_market_cap > 0:
        mcap_weighted_vol_change = weighted_vol_change / total_market_cap
    
    # 3. Net Money Flow
    net_flow = up_turnover - down_turnover
    total_flow = up_turnover + down_turnover
    flow_ratio = (up_turnover / total_flow * 100) if total_flow > 0 else 50
    
    # 4. Intraday Volume Scaling with NON-LINEAR distribution (FIXED)
    # Use IST timezone for Indian market
    current_time = datetime.now(IST)
    market_open = current_time.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = current_time.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    
    intraday_factor = 1.0
    expected_vol_factor = 1.0
    scaled_volume_pct_change = volume_pct_change
    
    if market_open <= current_time <= market_close:
        # Get expected volume % based on U-shaped distribution
        expected_vol_factor = get_expected_volume_factor(current_time)
        intraday_factor = expected_vol_factor
        
        if expected_vol_factor > 0:
            # Project full-day volume using non-linear curve
            projected_volume = total_volume_today / expected_vol_factor
            scaled_volume_ratio = projected_volume / total_volume_avg
            scaled_volume_pct_change = (scaled_volume_ratio - 1) * 100
    
    # Calculate "above average" count for backward compatibility
    stocks_above_avg_vol = stocks_elevated_vol + (stocks_normal_vol // 2)  # Include half of normal as "above"
    stocks_below_avg_vol = stocks_reduced_vol + (stocks_normal_vol - stocks_normal_vol // 2)
    
    # ========== ALERT LEVEL DETERMINATION ==========
    # Use volume ratio for clearer thresholds
    
    if volume_ratio >= 2.0:
        alert_level = "extreme_high"
        alert_text = "🔥 EXTREME VOLUME"
        alert_color = "#ff6600"  # Orange
        alert_bg = "linear-gradient(135deg, #ff6600 0%, #cc5200 100%)"
    elif volume_ratio >= 1.5:
        alert_level = "very_high"
        alert_text = "📈 VERY HIGH VOLUME"
        alert_color = "#00ff00"  # Bright green
        alert_bg = "linear-gradient(135deg, #00cc66 0%, #009944 100%)"
    elif volume_ratio >= 1.2:
        alert_level = "elevated"
        alert_text = "↗️ ELEVATED VOLUME"
        alert_color = "#88cc88"  # Light green
        alert_bg = "linear-gradient(135deg, #44aa44 0%, #338833 100%)"
    elif volume_ratio >= 0.8:
        alert_level = "normal"
        alert_text = "➡️ NORMAL VOLUME"
        alert_color = "#888888"  # Gray
        alert_bg = "linear-gradient(135deg, #555555 0%, #444444 100%)"
    elif volume_ratio >= 0.5:
        alert_level = "low"
        alert_text = "↘️ LOW VOLUME"
        alert_color = "#cc8888"  # Light red
        alert_bg = "linear-gradient(135deg, #aa4444 0%, #883333 100%)"
    else:
        alert_level = "very_low"
        alert_text = "📉 VERY LOW VOLUME"
        alert_color = "#ff4d4d"  # Red
        alert_bg = "linear-gradient(135deg, #cc3333 0%, #aa2222 100%)"
    
    # Determine flow direction
    if flow_ratio >= 60:
        flow_text = "🟢 STRONG BUYING"
        flow_color = "#00ff00"
    elif flow_ratio >= 52:
        flow_text = "🟢 BUYING"
        flow_color = "#00cc66"
    elif flow_ratio >= 48:
        flow_text = "⚪ NEUTRAL"
        flow_color = "#888888"
    elif flow_ratio >= 40:
        flow_text = "🔴 SELLING"
        flow_color = "#ff6666"
    else:
        flow_text = "🔴 STRONG SELLING"
        flow_color = "#ff0000"
    
    return {
        # Core metrics
        "total_turnover_today": total_turnover_today,
        "total_turnover_avg": total_turnover_avg,
        "volume_ratio": volume_ratio,
        "volume_pct_change": volume_pct_change,
        "scaled_volume_pct_change": scaled_volume_pct_change,
        "intraday_factor": intraday_factor,
        
        # Market cap weighted
        "mcap_weighted_vol_change": mcap_weighted_vol_change,
        
        # Directional flow
        "up_turnover": up_turnover,
        "down_turnover": down_turnover,
        "net_flow": net_flow,
        "flow_ratio": flow_ratio,
        "flow_text": flow_text,
        "flow_color": flow_color,
        
        # Market breadth
        "stocks_above_avg_vol": stocks_above_avg_vol,
        "stocks_below_avg_vol": stocks_below_avg_vol,
        "stocks_up": stocks_up,
        "stocks_down": stocks_down,
        "stocks_neutral": stocks_neutral,
        "valid_stocks": valid_stocks,
        
        # Alert info
        "alert_level": alert_level,
        "alert_text": alert_text,
        "alert_color": alert_color,
        "alert_bg": alert_bg,
    }


def format_turnover_crores(value):
    """Format turnover value in Crores for display."""
    if value is None:
        return "-"
    try:
        crores = value / 1e7  # Convert to Crores
        if crores >= 1000:
            return f"₹{crores/1000:,.2f}K Cr"
        return f"₹{crores:,.0f} Cr"
    except:
        return "-"
# -------------------------------------------------------------------
# 7) WEEKLY TURNOVER BAR CHART
# -------------------------------------------------------------------
def create_weekly_turnover_chart(stocks_data):
    """
    Create a bar chart showing aggregate turnover for the last 7 days.
    Uses existing data from stocks_data - no additional API calls.
    """
    if not stocks_data:
        return None
    
    # Aggregate turnover by date across all stocks
    daily_turnover = {}
    stocks_with_data = 0
    
    for stock in stocks_data:
        weekly_data = stock.get("WEEKLY_TURNOVER", [])
        if weekly_data:
            stocks_with_data += 1
        for day_data in weekly_data:
            date = day_data.get("date")
            turnover = day_data.get("turnover", 0)
            if date:
                daily_turnover[date] = daily_turnover.get(date, 0) + turnover
    
    print(f"DEBUG: {stocks_with_data}/{len(stocks_data)} stocks have WEEKLY_TURNOVER data. Daily turnover entries: {len(daily_turnover)}")
    
    if not daily_turnover:
        return None
    
    try:
        # Sort by date
        sorted_dates = sorted(daily_turnover.keys())
        dates = sorted_dates[-7:]  # Last 7 days
        turnovers = [daily_turnover[d] / 1e7 for d in dates]  # Convert to Crores
        
        print(f"DEBUG: Creating chart with dates={dates}, turnovers={turnovers}")
        
        # Format dates for display (e.g., "Mon 09")
        display_dates = []
        for d in dates:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
                display_dates.append(dt.strftime("%a %d"))
            except:
                display_dates.append(d[-5:])
        
        # Create bar chart
        fig = go.Figure()
        
        text_colors = ['#FFF' if t < (max(turnovers) * 0.35) else '#111' for t in turnovers]

        fig.add_trace(go.Bar(
            x=display_dates,
            y=turnovers,
            marker=dict(
                color=turnovers,
                colorscale=[[0, '#0099CC'], [0.5, '#00D4FF'], [1, '#00FF88']],
                line=dict(color='rgba(0, 212, 255, 0.8)', width=1)
            ),
            text=[f"₹{t:,.0f} Cr" for t in turnovers],
            textposition='auto',
            textfont=dict(color=text_colors, size=15),   # << updated
            hovertemplate="<b>%{x}</b><br>Turnover: ₹%{y:,.0f} Cr<extra></extra>"
        ))
        
        fig.update_layout(
            title=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#888"),
            height=220,
            margin=dict(l=50, r=20, t=50, b=40),
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='#333',
                tickfont=dict(size=11, color='#888')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(50,50,50,0.5)',
                showline=False,
                tickformat=',d',
                ticksuffix=' Cr',
                tickfont=dict(size=10, color='#666')
            ),
            hoverlabel=dict(
                bgcolor='#1a1a2e',
                font_size=12,
                font_family="Inter"
            )
        )
        
        chart = dcc.Graph(
            figure=fig,
            config={'displayModeBar': False},
            style={'width': '100%'}
        )
        print("DEBUG: Chart created successfully!")
        return chart
        
    except Exception as e:
        print(f"ERROR creating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


# -------------------------------------------------------------------
# 8) INDEX CANDLESTICK CHART
# -------------------------------------------------------------------
def create_index_candlestick_chart(index_name: str):
    """
    Create a professional candlestick chart for the selected index.
    Uses Yahoo Finance for OHLC data.
    
    Features:
    - 60-day daily candles for swing trading analysis
    - 20-day EMA overlay for trend identification
    - Volume bars at bottom
    - Professional dark theme styling
    
    Args:
        index_name: "NIFTY50" or "NIFTY100"
    
    Returns:
        dcc.Graph component or None if data unavailable
    """
    try:
        yf_ticker = INDEX_YF_TICKERS.get(index_name)
        if not yf_ticker:
            logger.warning(f"No Yahoo Finance ticker for {index_name}")
            return None
        
        # Fetch OHLC data
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period=f"{INDEX_CANDLE_DAYS + 10}d", interval="1d")
        
        if hist.empty or len(hist) < 10:
            logger.warning(f"Insufficient data for {index_name} candlestick chart")
            return None
        
        # Take last INDEX_CANDLE_DAYS
        hist = hist.tail(INDEX_CANDLE_DAYS)
        
        # Convert timezone-aware index to timezone-naive for Plotly compatibility
        hist = hist.reset_index()
        hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        # Format dates for cleaner x-axis labels (e.g., "Dec 14")
        hist['DateStr'] = hist['Date'].dt.strftime('%b %d')
        
        # Calculate 20-day EMA for trend
        hist['EMA20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        
        # Determine trend direction (last close vs EMA)
        current_close = hist['Close'].iloc[-1]
        current_ema = hist['EMA20'].iloc[-1]
        trend_up = current_close > current_ema
        
        # Create candlestick figure with subplots (candles + volume)
        fig = go.Figure()
        
        # Candlestick trace
        fig.add_trace(go.Candlestick(
            x=hist['DateStr'],
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=NSE_INDEX_NAMES.get(index_name, index_name),
            increasing=dict(line=dict(color='#00CC66', width=1), fillcolor='#00CC66'),
            decreasing=dict(line=dict(color='#FF4D4D', width=1), fillcolor='#FF4D4D'),
            hoverinfo='x+y+text',
        ))
        
        # 20-day EMA line
        fig.add_trace(go.Scatter(
            x=hist['DateStr'],
            y=hist['EMA20'],
            name='EMA 20',
            line=dict(color='#FFB800', width=2, dash='dot'),
            hovertemplate='EMA20: %{y:,.2f}<extra></extra>'
        ))
        
        # Get display name and calculate change
        display_name = NSE_INDEX_NAMES.get(index_name, index_name)
        price_change = current_close - hist['Close'].iloc[0]
        price_change_pct = (price_change / hist['Close'].iloc[0]) * 100
        change_color = "#00CC66" if price_change >= 0 else "#FF4D4D"
        change_symbol = "▲" if price_change >= 0 else "▼"
        
        # Update layout with professional styling
        fig.update_layout(
            title=dict(
                text=f"<b>{display_name}</b> <span style='font-size:14px;color:{change_color}'>{change_symbol} {price_change_pct:.2f}% ({INDEX_CANDLE_DAYS}D)</span>",
                font=dict(size=16, color='#fff'),
                x=0.02,
                xanchor='left'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#888"),
            height=320,
            margin=dict(l=60, r=20, t=50, b=40),
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='#333',
                tickfont=dict(size=10, color='#666'),
                rangeslider=dict(visible=False),
                type='category',  # Use category to remove weekend/holiday gaps
                nticks=10,        # Show ~10 date labels to avoid clutter
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(50,50,50,0.5)',
                showline=False,
                tickformat=',.0f',
                tickfont=dict(size=10, color='#666'),
                side='right',
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=10, color='#888'),
                bgcolor='rgba(0,0,0,0)'
            ),
            hoverlabel=dict(
                bgcolor='#1a1a2e',
                font_size=12,
                font_family="Inter"
            ),
            xaxis_rangeslider_visible=False,
        )
        
        # Add current price annotation
        fig.add_annotation(
            x=hist['DateStr'].iloc[-1],
            y=current_close,
            text=f"₹{current_close:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=change_color,
            ax=40,
            ay=0,
            font=dict(size=12, color=change_color, family="Inter"),
            bgcolor='rgba(26,26,46,0.9)',
            bordercolor=change_color,
            borderwidth=1,
            borderpad=4
        )
        
        chart = dcc.Graph(
            figure=fig,
            config={'displayModeBar': False},
            style={'width': '100%'}
        )
        
        logger.info(f"✓ Created candlestick chart for {index_name}")
        return chart
        
    except Exception as e:
        logger.error(f"Error creating candlestick chart for {index_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# -------------------------------------------------------------------
# 9) SECTOR ROTATION TRACKER (MCAP-Weighted)
# -------------------------------------------------------------------

# Emoji mapping for major NSE macro sectors
SECTOR_EMOJIS = {
    "Financial Services": "🏦",
    "Information Technology": "💻",
    "Fast Moving Consumer Goods": "🛒",
    "Energy": "⚡",
    "Healthcare": "💊",
    "Automobile and Auto Components": "🚗",
    "Consumer Services": "🎯",
    "Capital Goods": "🏭",
    "Metals & Mining": "⛏️",
    "Construction Materials": "🏗️",
    "Telecommunication": "📡",
    "Services": "🔧",
    "Chemicals": "🧪",
    "Consumer Durables": "📺",
    "Oil Gas & Consumable Fuels": "🛢️",
    "Diversified": "🎲",
    "Realty": "🏠",
    "Media Entertainment & Publication": "🎬",
    "Textiles": "👔",
    "Power": "⚡",
    "Others": "📦",
}


def calculate_sector_indices(stocks_data: list) -> list:
    """
    Calculate MCAP-weighted index change for each major sector.
    
    Uses ONLY existing data from stocks_data - NO additional API calls.
    
    Formula (NSE Standard):
    Sector Change % = Σ(Stock MCAP × Stock %Change) / Σ(Stock MCAP)
    
    Returns: List of sector dicts sorted by 1-day performance
    """
    if not stocks_data:
        return []
    
    # Group stocks by macro sector
    sectors = {}
    for stock in stocks_data:
        macro = stock.get("MACRO_SECTOR", "Others")
        if not macro or macro == "N/A":
            macro = "Others"
            
        if macro not in sectors:
            sectors[macro] = {
                "stocks": [],
                "total_mcap": 0,
                "weighted_change": 0,
                "total_turnover": 0,
                "stocks_up": 0,
                "stocks_down": 0,
            }
        
        mcap = safe_float(stock.get("MARKET_CAP_CR"))
        change_pct = safe_float(stock.get("TODAY_CURRENT_PRICE_CHANGE_PCT"))
        turnover = safe_float(stock.get("TODAYS_TURNOVER"))
        
        if mcap and mcap > 0:
            sectors[macro]["total_mcap"] += mcap
            if change_pct is not None:
                sectors[macro]["weighted_change"] += mcap * change_pct
                if change_pct > 0:
                    sectors[macro]["stocks_up"] += 1
                elif change_pct < 0:
                    sectors[macro]["stocks_down"] += 1
            sectors[macro]["stocks"].append(stock)
        
        if turnover:
            sectors[macro]["total_turnover"] += turnover
    
    # Calculate sector index changes
    result = []
    for sector_name, data in sectors.items():
        if len(data["stocks"]) < 2:  # Skip sectors with < 2 stocks
            continue
        
        if data["total_mcap"] > 0:
            sector_change_pct = data["weighted_change"] / data["total_mcap"]
        else:
            sector_change_pct = 0
        
        # Find top gainers and losers
        sorted_stocks = sorted(
            data["stocks"],
            key=lambda x: safe_float(x.get("TODAY_CURRENT_PRICE_CHANGE_PCT")) or 0,
            reverse=True
        )
        top_gainers = [s["SYMBOL"] for s in sorted_stocks[:2] if safe_float(s.get("TODAY_CURRENT_PRICE_CHANGE_PCT", 0)) > 0]
        top_losers = [s["SYMBOL"] for s in sorted_stocks[-2:] if safe_float(s.get("TODAY_CURRENT_PRICE_CHANGE_PCT", 0)) < 0]
        
        result.append({
            "sector": sector_name,
            "emoji": SECTOR_EMOJIS.get(sector_name, "📊"),
            "1d_change": round(sector_change_pct, 2),
            "total_turnover": data["total_turnover"],
            "stock_count": len(data["stocks"]),
            "total_mcap": data["total_mcap"],
            "stocks_up": data["stocks_up"],
            "stocks_down": data["stocks_down"],
            "top_gainers": top_gainers,
            "top_losers": top_losers,
        })
    
    # Sort by 1-day change (best performing first)
    result.sort(key=lambda x: x["1d_change"], reverse=True)
    return result


def create_sector_rotation_panel(sector_indices: list):
    """
    Create a visual panel showing sector rotation with MCAP-weighted changes.
    Uses existing glassmorphism styling for consistency.
    
    Returns: dbc.Card component or None if no data
    """
    if not sector_indices:
        return None
    
    sector_cards = []
    for i, sector in enumerate(sector_indices):
        change = sector["1d_change"]
        is_positive = change >= 0
        is_top = i == 0  # Best performing sector
        is_bottom = i == len(sector_indices) - 1  # Worst performing sector
        
        # Gradient based on performance intensity
        if change >= 1.5:
            bg_gradient = "linear-gradient(135deg, rgba(0,255,136,0.2) 0%, rgba(0,153,68,0.15) 100%)"
            border_color = "#00ff88"
        elif change >= 0.5:
            bg_gradient = "linear-gradient(135deg, rgba(0,204,102,0.15) 0%, rgba(0,102,51,0.1) 100%)"
            border_color = "#00cc66"
        elif change >= 0:
            bg_gradient = "linear-gradient(135deg, rgba(136,204,136,0.1) 0%, rgba(68,136,68,0.08) 100%)"
            border_color = "#88cc88"
        elif change >= -0.5:
            bg_gradient = "linear-gradient(135deg, rgba(204,136,136,0.1) 0%, rgba(136,68,68,0.08) 100%)"
            border_color = "#cc8888"
        elif change >= -1.5:
            bg_gradient = "linear-gradient(135deg, rgba(255,77,77,0.15) 0%, rgba(153,0,0,0.1) 100%)"
            border_color = "#ff4d4d"
        else:
            bg_gradient = "linear-gradient(135deg, rgba(255,51,51,0.2) 0%, rgba(170,0,0,0.15) 100%)"
            border_color = "#ff3333"
        
        # Top/bottom performer badges
        badge = None
        if is_top:
            badge = html.Span("🔥", style={"position": "absolute", "top": "-8px", "right": "-8px", "fontSize": "1rem"})
        elif is_bottom:
            badge = html.Span("❄️", style={"position": "absolute", "top": "-8px", "right": "-8px", "fontSize": "1rem"})
        
        # Turnover display in Crores
        turnover_cr = sector["total_turnover"] / CRORE_DIVISOR if sector["total_turnover"] else 0
        if turnover_cr >= 1000:
            turnover_display = f"₹{turnover_cr/1000:.1f}K Cr"
        else:
            turnover_display = f"₹{turnover_cr:.0f} Cr"
        
        # Breadth indicator (up vs down)
        breadth_ratio = sector["stocks_up"] / sector["stock_count"] * 100 if sector["stock_count"] > 0 else 50
        
        card_content = html.Div([
            badge,
            # Sector icon and name
            html.Div([
                html.Span(sector["emoji"], style={"fontSize": "2.2rem"}),
            ], style={"textAlign": "center", "marginBottom": "10px"}),
            
            html.Div(sector["sector"], style={
                "fontSize": "0.85rem", 
                "color": "#ddd", 
                "textAlign": "center",
                "height": "40px",
                "overflow": "hidden", 
                "textOverflow": "ellipsis",
                "lineHeight": "1.3",
                "fontWeight": "600",
            }),
            
            # Change percentage (prominent)
            html.Div([
                html.Span("▲ " if is_positive else "▼ ", style={"fontSize": "1.1rem"}),
                html.Span(f"{change:+.2f}%", style={
                    "fontSize": "1.5rem", 
                    "fontWeight": "700",
                    "color": "#00ff88" if change >= 1 else "#00cc66" if is_positive else "#ff4d4d" if change <= -1 else "#cc8888"
                })
            ], style={"textAlign": "center", "marginBottom": "12px"}),
            
            # Stock count and breadth
            html.Div([
                html.Span(f"{sector['stock_count']} ", style={"fontWeight": "600", "color": "#aaa"}),
                html.Span("stocks", style={"color": "#888"})
            ], style={"fontSize": "0.8rem", "textAlign": "center", "marginBottom": "8px"}),
            
            # Breadth bar
            html.Div([
                html.Div(style={
                    "width": f"{breadth_ratio}%",
                    "height": "5px",
                    "background": "linear-gradient(90deg, #00cc66, #00ff88)",
                    "borderRadius": "3px",
                }),
            ], style={
                "width": "100%",
                "height": "5px",
                "background": "#333",
                "borderRadius": "3px",
                "marginTop": "8px",
            }),
            
            # Turnover
            html.Div(turnover_display, style={
                "fontSize": "0.75rem", 
                "color": "#777", 
                "textAlign": "center",
                "marginTop": "10px",
                "fontWeight": "500",
            }),
            
            # Click hint
            html.Div("Click for details", style={
                "fontSize": "0.6rem",
                "color": "#555",
                "textAlign": "center",
                "marginTop": "8px",
                "fontStyle": "italic",
            })
        ], style={
            "background": bg_gradient,
            "border": f"1px solid {border_color}60",
            "borderRadius": "14px",
            "padding": "18px 14px",
            "minWidth": "145px",
            "maxWidth": "165px",
            "position": "relative",
        })
        
        # Wrap in clickable button
        card = html.Button(
            card_content,
            id={"type": "sector-card", "sector": sector["sector"]},
            style={
                "background": "transparent",
                "border": "none",
                "padding": "0",
                "cursor": "pointer",
                "transition": "transform 0.2s, box-shadow 0.2s",
                "flex": "0 0 auto",
            },
            className="sector-card-btn"
        )
        sector_cards.append(card)
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Div([
                    html.Span("MCAP-Weighted", style={
                        "fontSize": "0.7rem", 
                        "color": "#9b59b6", 
                        "fontWeight": "600",
                        "padding": "4px 10px",
                        "background": "rgba(155, 89, 182, 0.15)",
                        "borderRadius": "12px",
                        "border": "1px solid rgba(155, 89, 182, 0.3)",
                    })
                ])
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
        ], style={
            "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
            "borderBottom": "2px solid #9b59b6",
            "padding": "14px 20px"
        }),
        dbc.CardBody([
            # Sector cards row
            html.Div(sector_cards, style={
                "display": "flex",
                "gap": "12px",
                "overflowX": "auto",
                "padding": "8px 4px",
                "scrollbarWidth": "thin",
            }),
            # Sector detail placeholder - will be populated by callback
            html.Div(
                id="sector-detail-container",
                className="sector-detail-container",
                style={"marginTop": "0"}  # Will animate when content is added
            )
        ], style={"padding": "16px"})
    ], style={
        "background": "linear-gradient(135deg, #12121c 0%, #1a1a2e 50%, #0f0f1a 100%)",
        "border": "1px solid #333",
        "borderRadius": "15px",
        "boxShadow": "0 8px 32px rgba(155, 89, 182, 0.12), inset 0 1px 0 rgba(255,255,255,0.05)",
        "backdropFilter": "blur(10px)",
    })


# -------------------------------------------------------------------
# 10) FETCH DATA FOR SYMBOLS IN SELECTED INDUSTRY
# -------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_stocks_data_for_industry(symbols, days_comparison=10, batch_size: int = 50, workers: int = 5):
    """Fetch data for multiple stocks in parallel."""
    all_data = []

    def _fetch_one(symbol):
        try:
            fund = get_fundamentals(symbol)
            vol = get_volume_stats(symbol)
            hist_price, hist_change, hist_change_pct = get_historical_comparison(symbol, days_comparison)
            
            if not fund:
                return None

            price_info = safe_dict(fund.get('priceInfo', {}))
            last_price = fund.get('lastPrice')
            prev_close = safe_float(price_info.get('previousClose') or price_info.get('close'))
            today_open = safe_float(price_info.get('open'))

            price_change = None
            price_change_pct = None
            try:
                if last_price is not None and prev_close is not None:
                    price_change = safe_float(last_price)
                    if price_change is not None:
                        price_change = price_change - prev_close
                        if prev_close != 0:
                            price_change_pct = (price_change / prev_close) * 100
            except Exception:
                price_change = None
                price_change_pct = None

            stock_data = {
                "SYMBOL": symbol,
                "STOCK_NAME": symbol,
                "INDUSTRIES": fund.get('Sector', 'N/A'),
                "MACRO_SECTOR": fund.get('Macro', 'Others'),  # For sector rotation tracker
                "LAST_DAY_CLOSING_PRICE": prev_close,
                "TODAY_PRICE_OPEN": today_open,
                "TODAY_CURRENT_PRICE": last_price,
                "TODAY_CURRENT_PRICE_CHANGE": price_change,
                "TODAY_CURRENT_PRICE_CHANGE_PCT": price_change_pct,
                "HISTORICAL_PRICE": hist_price,
                "HISTORICAL_CHANGE": hist_change,
                "HISTORICAL_CHANGE_PCT": hist_change_pct,
                "TODAY_VOLUME_AVERAGE": vol.get('avg_volume'),
                "TODAY_VOLUME": vol.get('todays_volume'),
                "VOL_CHANGE_PCT": vol.get('volume_change_pct'),
                "WEEKLY_TURNOVER": vol.get('weekly_turnover', []),
                # NEW: Consistent turnover from Yahoo Finance (matches weekly chart)
                "TODAYS_TURNOVER": vol.get('todays_turnover'),
                "AVG_TURNOVER": vol.get('avg_turnover'),
                "MARKET_CAP_CR": fund.get('Market Cap'),
                "PE": fund.get('P/E'),
                "EPS": fund.get('EPS'),
                "52WEEK_HIGH": fund.get('52W High'),
                "52WEEK_LOW": fund.get('52W Low'),
            }
            return stock_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    symbols = list(symbols)
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        print(f"\n[Batch {batch_num}/{total_batches}] Fetching {len(batch)} symbols...")
        
        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as ex:
            futures = {ex.submit(_fetch_one, s): s for s in batch}
            for fut in as_completed(futures):
                res = None
                try:
                    res = fut.result()
                except Exception as e:
                    s = futures.get(fut)
                    print(f"Executor error for {s}: {e}")
                if res:
                    all_data.append(res)
        
        if i + batch_size < len(symbols):
            print(f"  Batch {batch_num} complete. Waiting 2s before next batch...")
            time.sleep(2)

    return all_data


# -------------------------------------------------------------------
# 6) DASH APP SETUP
# -------------------------------------------------------------------
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Turnexia | Stock Screener"
)

server = app.server
# Add custom CSS for dropdown styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            /* ========== BASE STYLES ========== */
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
                min-height: 100vh;
            }
            
            /* ========== DROPDOWN STYLES ========== */
            .Select-control {
                background-color: #2a2a2a !important;
                border: 1px solid #444 !important;
                transition: all 0.3s ease;
            }
            .Select-control:hover {
                border-color: #00D4FF !important;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
            }
            .Select-menu-outer {
                background-color: #2a2a2a !important;
                border: 1px solid #444 !important;
                z-index: 9999 !important;
            }
            .Select-option {
                background-color: #2a2a2a !important;
                color: #fff !important;
                padding: 10px 12px !important;
                transition: all 0.2s ease;
            }
            .Select-option:hover {
                background: linear-gradient(90deg, #00D4FF 0%, #0099CC 100%) !important;
                color: #000 !important;
                transform: translateX(5px);
            }
            .Select-value-label {
                color: #fff !important;
            }
            .Select-placeholder {
                color: #aaa !important;
            }
            .Select-input > input {
                color: #fff !important;
            }
            .is-focused .Select-control {
                border-color: #00D4FF !important;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
            }
            
            /* ========== TABLE STYLES ========== */
            .table-dark {
                background: transparent !important;
            }
            
            .table-dark thead {
                position: sticky;
                top: 0;
                z-index: 100;
            }
            
            .table-dark thead th {
                background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%) !important;
                border: none !important;
            }
            
            .table-dark tbody tr {
                transition: all 0.3s ease;
                border-bottom: 1px solid #2a2a2a;
            }
            
            .table-dark tbody tr:hover {
                transform: scale(1.005);
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);
                z-index: 10;
                position: relative;
            }
            
            .table-dark tbody tr:hover td {
                background: rgba(0, 212, 255, 0.08) !important;
            }
            
            /* Gainer row highlight */
            .gainer-row td {
                background: rgba(0, 204, 102, 0.08) !important;
            }
            .gainer-row:hover td {
                background: rgba(0, 204, 102, 0.15) !important;
            }
            
            /* Loser row highlight */
            .loser-row td {
                background: rgba(255, 77, 77, 0.08) !important;
            }
            .loser-row:hover td {
                background: rgba(255, 77, 77, 0.15) !important;
            }
            
            /* ========== BUTTON STYLES ========== */
            button, .btn {
                transition: all 0.3s ease !important;
            }
            
            button:hover, .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
            }
            
            button:active, .btn:active {
                transform: translateY(0);
            }
            
            /* ========== CARD STYLES ========== */
            .card {
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }
            
            /* ========== ANIMATIONS ========== */
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.02); }
            }
            
            @keyframes glow {
                0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
                50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .pulse-animation {
                animation: pulse 2s ease-in-out infinite;
            }
            
            .glow-animation {
                animation: glow 2s ease-in-out infinite;
            }
            
            /* ========== SCROLLBAR STYLES ========== */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #1a1a1a;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #00E5FF 0%, #00AADD 100%);
            }
            
            /* ========== LOADING STYLES ========== */
            ._dash-loading {
                background: rgba(0, 0, 0, 0.7) !important;
            }
            
            /* ========== NUMBER DISPLAY ========== */
            .big-number {
                font-variant-numeric: tabular-nums;
                letter-spacing: -0.5px;
            }
            
            /* ========== RESPONSIVE ========== */
            @media (max-width: 1200px) {
                .table-dark { font-size: 0.8rem; }
            }
            
            /* ========== ACCORDION STYLES ========== */
            .accordion {
                border-radius: 15px !important;
                overflow: hidden;
            }
            
            .accordion-item {
                background: transparent !important;
                border: 1px solid #333 !important;
                margin-bottom: 10px;
                border-radius: 12px !important;
                overflow: hidden;
            }
            
            .accordion-button {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
                color: #fff !important;
                font-weight: 700 !important;
                font-size: 1rem !important;
                padding: 15px 20px !important;
                border: none !important;
                box-shadow: none !important;
                transition: all 0.3s ease !important;
            }
            
            .accordion-button:not(.collapsed) {
                background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%) !important;
                color: #00D4FF !important;
                border-bottom: 2px solid #00D4FF !important;
            }
            
            .accordion-button:hover {
                background: linear-gradient(135deg, #1f2a3e 0%, #1d2138 100%) !important;
            }
            
            .accordion-button::after {
                filter: invert(1) brightness(2);
                transition: transform 0.3s ease !important;
            }
            
            .accordion-button:focus {
                box-shadow: none !important;
                border-color: #00D4FF !important;
            }
            
            .accordion-body {
                background: transparent !important;
                padding: 0 !important;
            }
            
            .accordion-collapse {
                transition: all 0.3s ease !important;
            }
            
            /* Sector card button hover effect */
            .sector-card-btn:hover {
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 8px 25px rgba(155, 89, 182, 0.3);
            }
            
            .sector-card-btn:active {
                transform: translateY(0) scale(0.98);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    
    # Enhanced Header with Logo
    html.Div([
        # Row 1: Logo centered
        html.Div([
            html.Img(
                src="/assets/logo.png",
                style={
                    "height": "50px",
                    "filter": "drop-shadow(0 0 20px rgba(0, 212, 255, 0.5))"
                }
            ),
        ], style={"textAlign": "center", "marginBottom": "10px"}),
        # Row 2: Index badge + tagline
        html.Div([
            html.Span(id="header-index-name", children="NIFTY 100", 
                     style={
                         "color": "#00D4FF",
                         "fontWeight": "500",
                         "fontSize": "0.75rem",
                         "padding": "5px 14px",
                         "borderRadius": "20px",
                         "background": "rgba(0, 212, 255, 0.1)",
                         "border": "1px solid rgba(0, 212, 255, 0.3)"
                     })
        ], style={"textAlign": "center", "marginBottom": "8px"}),
        html.P("Real-time stock screener with aggregate volume analytics", 
               style={"color": "#666", "fontSize": "0.85rem", "margin": "0", "letterSpacing": "0.5px", "textAlign": "center"})
    ], style={"padding": "15px 0 20px 0"}),
    
    dbc.Row([
        # NEW: Index Selector Dropdown
        dbc.Col([
            html.Label("Select Index:", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "5px"}),
            dcc.Dropdown(
                id="index-selector",
                options=[
                    {"label": f"{info['emoji']} {info['name']} ({info['count']} stocks)", "value": key}
                    for key, info in SUPPORTED_INDICES.items()
                ],
                value=DEFAULT_INDEX,
                clearable=False,
                style={
                    "backgroundColor": "#2a2a2a",
                    "color": "#fff",
                    "borderRadius": "5px",
                },
                className="custom-dropdown"
            )
        ], width=2),
        dbc.Col([
            html.Label("Select Industry:", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "5px"}),
            dcc.Dropdown(
                id="industry-filter",
                options=[],
                value=None,
                placeholder="Select an industry to load data...",
                style={
                    "backgroundColor": "#2a2a2a",
                    "color": "#fff",
                    "borderRadius": "5px",
                },
                className="custom-dropdown"
            )
        ], width=3),
        dbc.Col([
            html.Label("Days for Comparison:", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "5px"}),
            dcc.Input(
                id="days-input",
                type="number",
                placeholder="Enter days (e.g., 10, 50, 200)",
                value=10,
                min=1,
                max=365,
                style={
                    "backgroundColor": "#2a2a2a",
                    "color": "#fff",
                    "border": "1px solid #444",
                    "borderRadius": "5px",
                    "padding": "8px",
                    "width": "100%"
                }
            )
        ], width=2),
        dbc.Col([
            dbc.Button(
                "⟳ Refresh Data",
                id="refresh-btn",
                color="info",
                size="lg",
                disabled=True,
                style={
                    "fontWeight": "700",
                    "borderRadius": "5px",
                    "padding": "8px 16px",
                    "fontSize": "1rem",
                    "border": "none",
                    "background": "linear-gradient(135deg, #00D4FF 0%, #0099CC 100%)",
                    "color": "#000",
                    "cursor": "pointer",
                    "marginTop": "28px"
                }
            )
        ], width=2),
        dbc.Col([
            html.Div(id="update-timestamp", style={"textAlign": "right", "color": "#00D4FF", "fontSize": "0.8rem", "fontWeight": "600", "marginTop": "35px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})
        ], width=3),
    ], className="mb-3"),
    
    # Auto-refresh interval (5 minutes = 300000 ms)
    dcc.Interval(
        id='auto-refresh-interval',
        interval=5*60*1000,  # 5 minutes in milliseconds
        n_intervals=0,
        disabled=True  # Will be enabled when industry is selected
    ),
    
    dcc.Store(id="symbol-industry-map", data={}),
    dcc.Store(id="stocks-data-store", data={}),
    dcc.Store(id="current-days", data=10),
    dcc.Store(id="sort-column", data=None),
    dcc.Store(id="sort-direction", data="asc"),
    dcc.Store(id="selected-index", data=DEFAULT_INDEX),  # Track selected index
    dcc.Store(id="selected-sector", data=None),  # Track clicked sector for drill-down
    
    # Hidden placeholder for close button (always exists so callback doesn't fail)
    html.Button(id="close-sector-detail", n_clicks=0, style={"display": "none"}),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-1",
                type="circle",
                color="#00D4FF",
                children=html.Div(
                    id="table-container", 
                    style={"overflowX": "auto", "minHeight": "200px"},
                    children=[
                        # Default welcome content (shown before callbacks run)
                        html.Div([
                            html.Div([
                                html.Img(
                                    src="/assets/logo_symbol.png",
                                    style={
                                        "height": "60px",
                                        "marginBottom": "15px",
                                        "display": "block",
                                        "margin": "0 auto 15px auto",
                                        "filter": "drop-shadow(0 0 15px rgba(0, 212, 255, 0.4))"
                                    }
                                ),
                                html.H2("Loading...", style={"color": "#fff", "fontWeight": "700", "fontSize": "1.8rem", "marginBottom": "10px"}),
                                html.P("Preparing your stock analytics dashboard", style={"color": "#888", "fontSize": "1rem"})
                            ], style={"textAlign": "center", "padding": "60px 20px"})
                        ], style={
                            "background": "linear-gradient(180deg, rgba(0,212,255,0.03) 0%, transparent 50%)",
                            "borderRadius": "15px",
                            "padding": "40px 20px",
                            "marginTop": "20px"
                        })
                    ]
                )
            )
        ], width=12),
    ])
    
], fluid=True, style={"backgroundColor": "#1a1a1a", "color": "#fff", "padding": "20px"})


# -------------------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------------------

# Callback 1: Load symbol-industry mapping when index changes
@app.callback(
    [Output("symbol-industry-map", "data"),
     Output("industry-filter", "options"),
     Output("industry-filter", "value"),
     Output("header-index-name", "children"),
     Output("selected-index", "data")],
    Input("index-selector", "value")
)
def load_index_data(selected_index):
    """Load symbol-industry mapping for selected index.
    Uses cached data for efficiency - only fetches from API once per hour.
    """
    if not selected_index:
        selected_index = DEFAULT_INDEX
    
    logger.info(f"Loading index data for {selected_index}...")
    symbol_industry_map = load_index_with_industries(selected_index)
    
    if not symbol_industry_map:
        logger.warning(f"WARNING: No data loaded for {selected_index}!")
        return {}, [], None, selected_index, selected_index
    
    # Extract industries for dropdown (handle both dict and string formats)
    industries = set()
    for val in symbol_industry_map.values():
        if isinstance(val, dict):
            # New format: get 'industry' field from dict
            ind = val.get('industry', 'N/A')
        else:
            # Legacy format: string
            ind = val
        if ind and ind != 'N/A':
            industries.add(ind)
    
    # Add "All" option at the beginning
    options = [{"label": "🌐 All Industries", "value": "ALL"}]
    options.extend([{"label": ind, "value": ind} for ind in sorted(industries)])
    
    # Get display name for header badge
    index_info = SUPPORTED_INDICES.get(selected_index, {"name": selected_index})
    header_text = index_info['name']
    
    logger.info(f"✓ Loaded {len(symbol_industry_map)} stocks for {selected_index} with {len(options)-1} industries")
    
    return symbol_industry_map, options, None, header_text, selected_index


# Callback 2: Enable/Disable refresh button and auto-refresh
@app.callback(
    [Output("refresh-btn", "disabled"),
     Output("auto-refresh-interval", "disabled")],
    Input("industry-filter", "value")
)
def toggle_refresh_and_interval(selected_industry):
    """Enable refresh button and auto-refresh only when an industry is selected."""
    is_disabled = selected_industry is None
    return is_disabled, is_disabled


# Callback 3: Fetch data when industry selected, refreshed manually, or auto-refreshed
@app.callback(
    [Output("stocks-data-store", "data"),
     Output("update-timestamp", "children"),
     Output("current-days", "data")],
    [Input("industry-filter", "value"),
     Input("refresh-btn", "n_clicks"),
     Input("auto-refresh-interval", "n_intervals"),
     Input("days-input", "value")],
    State("symbol-industry-map", "data"),
    running=[
        (Output("refresh-btn", "disabled"), True, False),
    ]
)
def fetch_industry_data(selected_industry, manual_clicks, auto_intervals, days_input, symbol_industry_map):
    """Fetch stock data for the selected industry or all industries."""
    
    if not selected_industry or not symbol_industry_map:
        return {}, "", 10
    
    # Use default 10 days if invalid input
    days_comparison = days_input if days_input and days_input > 0 else 10
    
    # Determine if this was triggered by manual refresh or auto-refresh
    ctx = dash.callback_context
    trigger_source = "Initial Load"
    
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "refresh-btn":
            trigger_source = "Manual Refresh"
        elif trigger_id == "auto-refresh-interval":
            trigger_source = "Auto Refresh"
        elif trigger_id == "days-input":
            trigger_source = "Days Changed"
    
    # Clear caches on manual refresh or auto refresh
    if manual_clicks or auto_intervals > 0 or trigger_source == "Days Changed":
        nse_data_cache.clear()
        fundamentals_cache.clear()
        volume_cache.clear()
        historical_cache.clear()
        logger.info(f"[{trigger_source.upper()}] Caches cleared at {datetime.now(IST).strftime('%H:%M:%S')}")
    
    # Get symbols for selected industry or all symbols
    if selected_industry == "ALL":
        symbols_in_industry = list(symbol_industry_map.keys())
        display_name = "All Industries"
    else:
        # Handle both dict and string format for industry mapping
        symbols_in_industry = []
        for symbol, industry_data in symbol_industry_map.items():
            if isinstance(industry_data, dict):
                # New format: compare with 'industry' field
                if industry_data.get('industry') == selected_industry:
                    symbols_in_industry.append(symbol)
            else:
                # Legacy format: string comparison
                if industry_data == selected_industry:
                    symbols_in_industry.append(symbol)
        display_name = selected_industry
    
    if not symbols_in_industry:
        return {}, f"No stocks found for {display_name}", days_comparison
    
    print(f"\n[FETCHING DATA - {trigger_source}] Loading {len(symbols_in_industry)} stocks for: {display_name} (comparing {days_comparison} days)")
    
    # Fetch data with historical comparison
    stocks_data = fetch_stocks_data_for_industry(symbols_in_industry, days_comparison=days_comparison)
    
    # Create timestamp with source indicator (shortened to fit in column)
    now = datetime.now().strftime("%H:%M:%S")
    timestamp = f"Last Updated: {now} | {len(stocks_data)} stocks | {days_comparison}D | Next ↻ in 5m"
    
    return {selected_industry: stocks_data}, timestamp, days_comparison


# Callback 4: Handle column sorting
@app.callback(
    [Output("sort-column", "data"),
     Output("sort-direction", "data")],
    Input({"type": "sort-button", "column": ALL}, "n_clicks"),
    [State("sort-column", "data"),
     State("sort-direction", "data")],
    prevent_initial_call=True
)
def handle_sort(n_clicks_list, current_column, current_direction):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_column, current_direction

    # Extract which button triggered
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    # SECURITY FIX: Use json.loads instead of eval
    triggered_id = json.loads(triggered)

    column = triggered_id["column"]

    # Toggle direction
    if column == current_column:
        new_direction = "desc" if current_direction == "asc" else "asc"
    else:
        new_direction = "desc"  # default

    return column, new_direction

# Callback 5: Handle sector card click
@app.callback(
    Output("selected-sector", "data", allow_duplicate=True),
    Input({"type": "sector-card", "sector": ALL}, "n_clicks"),
    State("selected-sector", "data"),
    prevent_initial_call=True
)
def handle_sector_click(n_clicks_list, current_sector):
    """Toggle sector selection when a card is clicked."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_sector
    
    # Check if any button was actually clicked (not just initialized)
    if not any(n_clicks_list):
        return current_sector
    
    # Extract which sector was clicked
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    triggered_id = json.loads(triggered)
    clicked_sector = triggered_id["sector"]
    
    # Toggle: if same sector clicked again, close it
    if clicked_sector == current_sector:
        logger.info(f"Sector panel closed: {clicked_sector}")
        return None
    
    logger.info(f"Sector selected: {clicked_sector}")
    return clicked_sector

# Callback 5b: Handle close button click
@app.callback(
    Output("selected-sector", "data", allow_duplicate=True),
    Input("close-sector-detail", "n_clicks"),
    prevent_initial_call=True
)
def handle_close_sector_detail(n_clicks):
    """Close sector detail panel when close button is clicked."""
    if n_clicks:
        logger.info("Sector panel closed via close button")
        return None
    return dash.no_update

# Callback 5c: Update sector detail panel content (separate from main table for smooth animation)
@app.callback(
    Output("sector-detail-container", "children"),
    [Input("selected-sector", "data"),
     Input("stocks-data-store", "data"),
     Input("current-days", "data")],
    prevent_initial_call=True
)
def update_sector_detail_panel(selected_sector, stocks_data_store, days):
    """
    Update the sector detail panel inside the sector rotation card.
    This callback only updates the sector detail content, not the entire page.
    Uses existing data from stocks_data_store - no additional API calls.
    """
    if not selected_sector or not stocks_data_store:
        return None  # Return empty - will collapse smoothly via CSS
    
    # Get current industry from stocks_data_store keys
    current_industry = list(stocks_data_store.keys())[0] if stocks_data_store else None
    if not current_industry:
        return None
    
    stocks_data = stocks_data_store.get(current_industry, [])
    if not stocks_data:
        return None
    
    days = days if days and days > 0 else 10
    
    # Filter stocks by MACRO_SECTOR - using existing data, no API calls!
    sector_stocks = [s for s in stocks_data if s.get("MACRO_SECTOR") == selected_sector]
    logger.info(f"Sector detail update: {selected_sector} has {len(sector_stocks)} stocks")
    
    if not sector_stocks:
        return html.Div(
            f"No stocks found in {selected_sector}",
            style={"color": "#888", "padding": "20px", "textAlign": "center"}
        )
    
    # Helper functions for formatting (same as generate_table)
    def format_value(val, decimals=2):
        if val is None:
            return "-"
        try:
            v = float(val)
            return f"{v:,.{decimals}f}"
        except:
            return str(val)

    def format_pct(val):
        if val is None:
            return "-"
        try:
            v = float(val)
            color = "#00cc66" if v >= 0 else "#ff4d4d"
            symbol = "▲" if v >= 0 else "▼"
            return html.Span(f"{symbol} {v:.2f}%", style={"color": color, "fontWeight": "700"})
        except:
            return str(val)

    def format_currency(val, decimals=2):
        if val is None:
            return "-"
        try:
            v = float(val)
            return f"₹{v:,.{decimals}f}"
        except:
            return str(val)

    def format_marketcap(val):
        if val is None:
            return "-"
        try:
            v = float(val)
            cr = v / 1e7
            return f"₹{cr:,.2f}Cr"
        except:
            return str(val)
    
    # Build full table with ALL indicators (matching main table)
    sector_rows = []
    for i, stock in enumerate(sector_stocks):
        row_bg = "#1a1a1a" if i % 2 == 0 else "#222222"
        
        # Determine row styling based on price change
        price_change_pct = stock.get("TODAY_CURRENT_PRICE_CHANGE_PCT")
        row_border_color = "transparent"
        if price_change_pct is not None:
            if price_change_pct > 2:
                row_border_color = "#00cc66"  # Strong gainer
            elif price_change_pct < -2:
                row_border_color = "#ff4d4d"  # Strong loser
        
        sector_rows.append(html.Tr([
            # S.No
            html.Td(i + 1, style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "center", "color": "#777", "minWidth": "40px"}),
            # SYMBOL
            html.Td(stock["SYMBOL"], style={"backgroundColor": row_bg, "padding": "8px", "fontWeight": "700", "color": "#00D4FF", "borderLeft": f"3px solid {row_border_color}", "minWidth": "80px"}),
            # INDUSTRIES
            html.Td(stock.get("INDUSTRIES", "-"), style={"backgroundColor": row_bg, "padding": "8px", "fontSize": "0.8rem", "color": "#aaa", "minWidth": "120px"}),
            # LAST CLOSE
            html.Td(format_currency(stock.get("LAST_DAY_CLOSING_PRICE")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "85px"}),
            # OPEN
            html.Td(format_currency(stock.get("TODAY_PRICE_OPEN")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "85px"}),
            # CURRENT
            html.Td(format_currency(stock.get("TODAY_CURRENT_PRICE")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "fontWeight": "700", "color": "#fff", "minWidth": "85px"}),
            # 1D CHANGE
            html.Td(format_currency(stock.get("TODAY_CURRENT_PRICE_CHANGE")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "80px"}),
            # 1D CHANGE %
            html.Td(format_pct(stock.get("TODAY_CURRENT_PRICE_CHANGE_PCT")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "90px"}),
            # 10D PRICE (dynamic days)
            html.Td(format_currency(stock.get("HISTORICAL_PRICE")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "fontSize": "0.8rem", "color": "#999", "minWidth": "85px"}),
            # 10D CHANGE
            html.Td(format_currency(stock.get("HISTORICAL_CHANGE")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "fontSize": "0.8rem", "color": "#999", "minWidth": "80px"}),
            # 10D CHANGE %
            html.Td(format_pct(stock.get("HISTORICAL_CHANGE_PCT")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "90px"}),
            # 52W HIGH
            html.Td(format_currency(stock.get("52WEEK_HIGH")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "color": "#888", "minWidth": "85px"}),
            # 52W LOW
            html.Td(format_currency(stock.get("52WEEK_LOW")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "color": "#888", "minWidth": "85px"}),
            # MARKET CAP (Cr)
            html.Td(format_marketcap(stock.get("MARKET_CAP_CR")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "100px"}),
            # P/E
            html.Td(format_value(stock.get("PE"), decimals=2), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "color": "#888", "minWidth": "60px"}),
            # EPS
            html.Td(format_value(stock.get("EPS"), decimals=2), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "color": "#888", "minWidth": "70px"}),
            # AVG VOLUME
            html.Td(format_value(stock.get("TODAY_VOLUME_AVERAGE"), decimals=0), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "fontSize": "0.8rem", "color": "#777", "minWidth": "90px"}),
            # TODAY VOLUME
            html.Td(format_value(stock.get("TODAY_VOLUME"), decimals=0), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "fontSize": "0.8rem", "color": "#777", "minWidth": "90px"}),
            # VOL CHANGE %
            html.Td(format_pct(stock.get("VOL_CHANGE_PCT")), style={"backgroundColor": row_bg, "padding": "8px", "textAlign": "right", "minWidth": "90px"}),
        ], className="sector-stock-row"))
    
    # Build header row with same columns as main table
    sector_header_style = {"padding": "10px 8px", "textAlign": "center", "color": "#000", "fontWeight": "700", "whiteSpace": "nowrap"}
    sector_table = html.Table([
        html.Thead(html.Tr([
            html.Th("S.No", style={**sector_header_style, "minWidth": "40px"}),
            html.Th("SYMBOL", style={**sector_header_style, "minWidth": "80px"}),
            html.Th("INDUSTRIES", style={**sector_header_style, "textAlign": "left", "minWidth": "120px"}),
            html.Th("LAST CLOSE", style={**sector_header_style, "textAlign": "right", "minWidth": "85px"}),
            html.Th("OPEN", style={**sector_header_style, "textAlign": "right", "minWidth": "85px"}),
            html.Th("CURRENT", style={**sector_header_style, "textAlign": "right", "minWidth": "85px"}),
            html.Th("1D CHANGE", style={**sector_header_style, "textAlign": "right", "minWidth": "80px"}),
            html.Th("1D CHANGE %", style={**sector_header_style, "textAlign": "right", "minWidth": "90px"}),
            html.Th(f"{days}D PRICE", style={**sector_header_style, "textAlign": "right", "minWidth": "85px"}),
            html.Th(f"{days}D CHANGE", style={**sector_header_style, "textAlign": "right", "minWidth": "80px"}),
            html.Th(f"{days}D CHANGE %", style={**sector_header_style, "textAlign": "right", "minWidth": "90px"}),
            html.Th("52W HIGH", style={**sector_header_style, "textAlign": "right", "minWidth": "85px"}),
            html.Th("52W LOW", style={**sector_header_style, "textAlign": "right", "minWidth": "85px"}),
            html.Th("MARKET CAP (Cr)", style={**sector_header_style, "textAlign": "right", "minWidth": "100px"}),
            html.Th("P/E", style={**sector_header_style, "textAlign": "right", "minWidth": "60px"}),
            html.Th("EPS", style={**sector_header_style, "textAlign": "right", "minWidth": "70px"}),
            html.Th("AVG VOLUME", style={**sector_header_style, "textAlign": "right", "minWidth": "90px"}),
            html.Th("TODAY VOLUME", style={**sector_header_style, "textAlign": "right", "minWidth": "90px"}),
            html.Th("VOL CHANGE %", style={**sector_header_style, "textAlign": "right", "minWidth": "90px"}),
        ], style={"background": "linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)"})),
        html.Tbody(sector_rows)
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"})
    
    # Get sector emoji from SECTOR_EMOJIS
    sector_emoji = SECTOR_EMOJIS.get(selected_sector, "📊")
    
    # Calculate sector change from stocks
    total_mcap = sum(s.get("MARKET_CAP_CR", 0) or 0 for s in sector_stocks)
    weighted_change = sum((s.get("MARKET_CAP_CR", 0) or 0) * (s.get("TODAY_CURRENT_PRICE_CHANGE_PCT", 0) or 0) for s in sector_stocks)
    sector_change = weighted_change / total_mcap if total_mcap > 0 else 0
    
    # Build the panel with animation class
    return html.Div([
        # Header with sector info
        html.Div([
            html.Div([
                html.Span(sector_emoji, style={"fontSize": "1.5rem", "marginRight": "12px"}),
                html.Span(f"{selected_sector}", style={"fontWeight": "700", "fontSize": "1.1rem", "color": "#fff"}),
                html.Span(f" ({len(sector_stocks)} stocks)", style={"color": "#888", "fontSize": "0.9rem", "marginLeft": "8px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Span(f"{'▲' if sector_change >= 0 else '▼'} {sector_change:+.2f}%", style={
                    "color": "#00ff88" if sector_change >= 0 else "#ff4d4d",
                    "fontWeight": "700",
                    "fontSize": "1.1rem",
                    "marginRight": "15px"
                }),
                html.Span("(click card to close)", style={
                    "color": "#666",
                    "fontSize": "0.75rem",
                    "fontStyle": "italic"
                })
            ], style={"display": "flex", "alignItems": "center"})
        ], style={
            "display": "flex", 
            "justifyContent": "space-between", 
            "alignItems": "center",
            "background": "linear-gradient(135deg, #2a1a3e 0%, #1a1a2e 100%)",
            "borderBottom": "2px solid #9b59b6",
            "padding": "15px 20px",
            "borderRadius": "10px 10px 0 0"
        }),
        # Table content
        html.Div(
            sector_table, 
            className="sector-detail-table",
            style={"maxHeight": "350px", "overflowY": "auto", "overflowX": "auto"}
        )
    ], className="slide-down-enter", style={
        "background": "linear-gradient(135deg, #15101e 0%, #1a1a2e 50%, #0f0f1a 100%)",
        "border": "2px solid #9b59b6",
        "borderRadius": "12px",
        "marginTop": "15px",
        "boxShadow": "0 8px 32px rgba(155, 89, 182, 0.25), inset 0 1px 0 rgba(255,255,255,0.05)",
    })

# Callback 6: Generate table with sorting
@app.callback(
    Output("table-container", "children"),
    [Input("stocks-data-store", "data"),
     Input("industry-filter", "value"),
     Input("current-days", "data"),
     Input("sort-column", "data"),
     Input("sort-direction", "data"),
     Input("selected-index", "data")]  # Removed selected-sector - handled by separate callback now
)
def generate_table(stocks_data_store, selected_industry, days, sort_column, sort_direction, selected_index):
    """Generate table with stock data, candlestick chart, and sorting."""

    if not selected_industry:
        return html.Div([
            # Welcome Section
            html.Div([
                html.Div([
                    html.Img(
                        src="/assets/logo_symbol.png",
                        style={
                            "height": "60px",
                            "marginBottom": "15px",
                            "display": "block",
                            "margin": "0 auto 15px auto",
                            "filter": "drop-shadow(0 0 15px rgba(0, 212, 255, 0.4))"
                        }
                    ),
                    html.H2("Get Started", style={
                        "color": "#fff",
                        "fontWeight": "700",
                        "fontSize": "1.8rem",
                        "marginBottom": "10px"
                    }),
                    html.P("Select an industry to explore real-time stock analytics", style={
                        "color": "#888",
                        "fontSize": "1rem",
                        "marginBottom": "25px"
                    }),
                    html.Div([
                        html.Span("👆 ", style={"fontSize": "1.2rem"}),
                        html.Span("Select an industry from the dropdown above to get started", style={
                            "background": "linear-gradient(90deg, #00D4FF, #00FF88)",
                            "backgroundClip": "text",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "fontSize": "1.1rem",
                            "fontWeight": "600"
                        })
                    ])
                ], style={"textAlign": "center", "padding": "40px 20px"}),
                
                # Feature Cards
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div("📊", style={"fontSize": "2rem", "marginBottom": "10px"}),
                                html.H5("Aggregate Volume", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "8px"}),
                                html.P("Track market-wide volume changes with price-weighted turnover analysis", 
                                      style={"color": "#888", "fontSize": "0.85rem", "margin": "0"})
                            ], style={
                                "background": "linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%)",
                                "border": "1px solid #333",
                                "borderRadius": "12px",
                                "padding": "25px",
                                "textAlign": "center",
                                "height": "100%"
                            })
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Div("🏛️", style={"fontSize": "2rem", "marginBottom": "10px"}),
                                html.H5("MCAP Weighted", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "8px"}),
                                html.P("See if large-cap stocks are leading or lagging the market movement", 
                                      style={"color": "#888", "fontSize": "0.85rem", "margin": "0"})
                            ], style={
                                "background": "linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%)",
                                "border": "1px solid #333",
                                "borderRadius": "12px",
                                "padding": "25px",
                                "textAlign": "center",
                                "height": "100%"
                            })
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Div("📈", style={"fontSize": "2rem", "marginBottom": "10px"}),
                                html.H5("Volume Breadth", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "8px"}),
                                html.P("Monitor how many stocks are trading above average volume", 
                                      style={"color": "#888", "fontSize": "0.85rem", "margin": "0"})
                            ], style={
                                "background": "linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%)",
                                "border": "1px solid #333",
                                "borderRadius": "12px",
                                "padding": "25px",
                                "textAlign": "center",
                                "height": "100%"
                            })
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Div("🔄", style={"fontSize": "2rem", "marginBottom": "10px"}),
                                html.H5("Flow Direction", style={"color": "#00D4FF", "fontWeight": "600", "marginBottom": "8px"}),
                                html.P("Track money flow into rising vs falling stocks in real-time", 
                                      style={"color": "#888", "fontSize": "0.85rem", "margin": "0"})
                            ], style={
                                "background": "linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%)",
                                "border": "1px solid #333",
                                "borderRadius": "12px",
                                "padding": "25px",
                                "textAlign": "center",
                                "height": "100%"
                            })
                        ], width=3),
                    ], className="g-3", style={"marginTop": "20px"})
                ], style={"padding": "0 40px"}),
                
                # Quick tip
                html.Div([
                    html.Hr(style={"borderColor": "#333", "margin": "40px 0 25px 0"}),
                    html.P([
                        html.Span("💡 Tip: ", style={"color": "#FFB800"}),
                        html.Span("Select ", style={"color": "#888"}),
                        html.Span("'🌐 All Industries'", style={"color": "#00D4FF", "fontWeight": "600"}),
                        html.Span(" to see the complete stocks aggregate volume indicator", style={"color": "#888"})
                    ], style={"textAlign": "center", "fontSize": "0.9rem"})
                ])
            ], style={
                "background": "linear-gradient(180deg, rgba(0,212,255,0.03) 0%, transparent 50%)",
                "borderRadius": "15px",
                "padding": "40px 20px",
                "marginTop": "20px"
            })
        ])

    stocks_data = stocks_data_store.get(selected_industry, [])
    
    # Debug: Check if WEEKLY_TURNOVER exists in the first stock
    if stocks_data:
        first_stock = stocks_data[0]
        wt = first_stock.get("WEEKLY_TURNOVER", "MISSING")
        print(f"DEBUG generate_table: {len(stocks_data)} stocks, first stock WEEKLY_TURNOVER = {type(wt).__name__}, len={len(wt) if isinstance(wt, list) else 'N/A'}")
    else:
        print("DEBUG generate_table: stocks_data is empty")

    if not stocks_data:
        return html.Div([
            html.P("No data loaded yet. Please wait...", 
                   style={"color": "#888", "fontSize": "1rem", "textAlign": "center", "marginTop": "50px"})
        ])

    # Use days from store (defaults to 10 if not set)
    days = days if days and days > 0 else 10
    
    # Sort data if sort column is specified
    if sort_column and stocks_data:
        # Map column names to data keys
        column_mapping = {
            "SYMBOL": "SYMBOL",
            "INDUSTRIES": "INDUSTRIES",
            "LAST_CLOSE": "LAST_DAY_CLOSING_PRICE",
            "OPEN": "TODAY_PRICE_OPEN",
            "CURRENT": "TODAY_CURRENT_PRICE",
            "1D_CHANGE": "TODAY_CURRENT_PRICE_CHANGE",
            "1D_CHANGE_PCT": "TODAY_CURRENT_PRICE_CHANGE_PCT",
            "ND_PRICE": "HISTORICAL_PRICE",
            "ND_CHANGE": "HISTORICAL_CHANGE",
            "ND_CHANGE_PCT": "HISTORICAL_CHANGE_PCT",
            "52W_HIGH": "52WEEK_HIGH",
            "52W_LOW": "52WEEK_LOW",
            "MARKET_CAP": "MARKET_CAP_CR",
            "PE": "PE",
            "EPS": "EPS",
            "AVG_VOLUME": "TODAY_VOLUME_AVERAGE",
            "TODAY_VOLUME": "TODAY_VOLUME",
            "VOL_CHANGE_PCT": "VOL_CHANGE_PCT",
        }
        
        data_key = column_mapping.get(sort_column)
        if data_key:
            # Sort with None values at the end
            reverse = (sort_direction == "desc")
            stocks_data = sorted(
                stocks_data,
                key=lambda x: (x.get(data_key) is None, x.get(data_key) if x.get(data_key) is not None else 0),
                reverse=reverse
            )
    
    def format_value(val, decimals=2):
        if val is None:
            return "-"
        try:
            v = float(val)
            return f"{v:,.{decimals}f}"
        except:
            return str(val)

    def format_pct(val):
        if val is None:
            return "-"
        try:
            v = float(val)
            color = "#00cc66" if v >= 0 else "#ff4d4d"
            symbol = "▲" if v >= 0 else "▼"
            return html.Span(f"{symbol} {v:.2f}%", style={"color": color, "fontWeight": "700"})
        except:
            return str(val)

    def format_currency(val, decimals=2):
        if val is None:
            return "-"
        try:
            v = float(val)
            return f"₹{v:,.{decimals}f}"
        except:
            return str(val)

    def format_marketcap(val):
        if val is None:
            return "-"
        try:
            v = float(val)
            cr = v / 1e7
            return f"₹{cr:,.2f}Cr"
        except:
            return str(val)

    # Build table
    rows = []
    
    # Helper function to create sortable header
    def create_header(label, column_key, align="center"):
        is_sorted = (sort_column == column_key)
        sort_indicator = " ▼" if sort_direction == "desc" else " ▲" if is_sorted else ""

        return html.Th(
            html.Button(
                label + sort_indicator,
                id={"type": "sort-button", "column": column_key},
                n_clicks=0,
                style={
                    "backgroundColor": "#00D4FF",
                    "color": "#000",
                    "fontWeight": "700",
                    "padding": "10px",
                    "textAlign": align,
                    "border": "none",
                    "cursor": "pointer",
                    "width": "100%",
                    "fontSize": "0.9rem"
                }
            ),
            style={"backgroundColor": "#00D4FF", "padding": "0"}
        )


    # Header - Added S.No as first column
    header_row = html.Tr([
        html.Th("S.No", style={"backgroundColor": "#00D4FF", "color": "#000", "fontWeight": "700", "padding": "10px", "textAlign": "center", "fontSize": "0.9rem"}),
        create_header("SYMBOL", "SYMBOL", "center"),
        create_header("INDUSTRIES", "INDUSTRIES", "left"),
        create_header("LAST CLOSE", "LAST_CLOSE", "right"),
        create_header("OPEN", "OPEN", "right"),
        create_header("CURRENT", "CURRENT", "right"),
        create_header("1D CHANGE", "1D_CHANGE", "right"),
        create_header("1D CHANGE %", "1D_CHANGE_PCT", "right"),
        create_header(f"{days}D PRICE", "ND_PRICE", "right"),
        create_header(f"{days}D CHANGE", "ND_CHANGE", "right"),
        create_header(f"{days}D CHANGE %", "ND_CHANGE_PCT", "right"),
        create_header("52W HIGH", "52W_HIGH", "right"),
        create_header("52W LOW", "52W_LOW", "right"),
        create_header("MARKET CAP (Cr)", "MARKET_CAP", "right"),
        create_header("P/E", "PE", "right"),
        create_header("EPS", "EPS", "right"),
        create_header("AVG VOLUME", "AVG_VOLUME", "right"),
        create_header("TODAY VOLUME", "TODAY_VOLUME", "right"),
        create_header("VOL CHANGE %", "VOL_CHANGE_PCT", "right"),
    ])
    rows.append(header_row)

    # Data rows
    for i, stock in enumerate(stocks_data):
        row_bg = "#2a2a2a" if i % 2 == 0 else "#1a1a1a"
        
        # Determine row class based on price change
        price_change = stock.get("TODAY_CURRENT_PRICE_CHANGE_PCT")
        row_class = ""
        if price_change is not None:
            if price_change > 2:  # Strong gainer (>2%)
                row_class = "gainer-row"
            elif price_change < -2:  # Strong loser (<-2%)
                row_class = "loser-row"
        
        # Add Serial Number (starting from 1)
        serial_no = i + 1
        row = html.Tr([
            html.Td(serial_no, style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "center", "fontWeight": "600", "color": "#888"}),
            html.Td(stock["SYMBOL"], style={"backgroundColor": row_bg, "padding": "10px 8px", "fontWeight": "700", "color": "#00D4FF", "borderLeft": "3px solid " + ("#00cc66" if row_class == "gainer-row" else "#ff4d4d" if row_class == "loser-row" else "transparent")}),
            html.Td(stock["INDUSTRIES"], style={"backgroundColor": row_bg, "padding": "10px 8px", "fontSize": "0.85rem", "color": "#aaa"}),
            html.Td(format_currency(stock["LAST_DAY_CLOSING_PRICE"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
            html.Td(format_currency(stock["TODAY_PRICE_OPEN"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
            html.Td(format_currency(stock["TODAY_CURRENT_PRICE"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "fontWeight": "700", "color": "#fff"}),
            html.Td(format_currency(stock["TODAY_CURRENT_PRICE_CHANGE"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
            html.Td(format_pct(stock["TODAY_CURRENT_PRICE_CHANGE_PCT"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
            html.Td(format_currency(stock.get("HISTORICAL_PRICE")), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "fontSize": "0.85rem", "color": "#999"}),
            html.Td(format_currency(stock.get("HISTORICAL_CHANGE")), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "fontSize": "0.85rem", "color": "#999"}),
            html.Td(format_pct(stock.get("HISTORICAL_CHANGE_PCT")), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
            html.Td(format_currency(stock["52WEEK_HIGH"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "color": "#888"}),
            html.Td(format_currency(stock["52WEEK_LOW"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "color": "#888"}),
            html.Td(format_marketcap(stock["MARKET_CAP_CR"]), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
            html.Td(format_value(stock["PE"], decimals=2), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "color": "#888"}),
            html.Td(format_value(stock.get("EPS"), decimals=2), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "color": "#888"}),
            html.Td(format_value(stock["TODAY_VOLUME_AVERAGE"], decimals=0), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "fontSize": "0.8rem", "color": "#777"}),
            html.Td(format_value(stock.get("TODAY_VOLUME"), decimals=0), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right", "fontSize": "0.8rem", "color": "#777"}),
            html.Td(format_pct(stock.get("VOL_CHANGE_PCT")), style={"backgroundColor": row_bg, "padding": "10px 8px", "textAlign": "right"}),
        ], className=row_class)
        rows.append(row)
    
    table = dbc.Table(
        html.Tbody(rows),
        bordered=False,
        className="table-dark",
        hover=True,
        responsive=True,
        style={"fontSize": "0.9rem", "marginTop": "20px"}
    )
    
    count_indicator = html.Div(
        f"Showing {len(stocks_data)} stocks" + (f" in {selected_industry}" if selected_industry != "ALL" else " across all industries"),
        style={"color": "#00D4FF", "fontSize": "0.95rem", "fontWeight": "600", "marginBottom": "10px"}
    )
    
    # ========== AGGREGATE VOLUME INDICATOR CARD ==========
    vol_indicator = calculate_aggregate_volume_indicator(stocks_data)
    
    # Create weekly turnover chart before building the card
    weekly_turnover_chart = create_weekly_turnover_chart(stocks_data)
    print(f"DEBUG weekly_turnover_chart is: {type(weekly_turnover_chart).__name__}, truthy={bool(weekly_turnover_chart)}")
    
    # ========== INDEX CANDLESTICK CHART ==========
    index_candlestick_chart = create_index_candlestick_chart(selected_index or DEFAULT_INDEX)
    print(f"DEBUG index_candlestick_chart: index={selected_index}, chart_type={type(index_candlestick_chart).__name__}, truthy={bool(index_candlestick_chart)}")
    
    # ========== SECTOR ROTATION TRACKER ==========
    sector_indices = calculate_sector_indices(stocks_data)
    sector_rotation_panel = create_sector_rotation_panel(sector_indices)
    print(f"DEBUG sector_rotation: {len(sector_indices)} sectors found")
    
    # NOTE: Sector detail panel is now handled by separate callback (update_sector_detail_panel)
    # which updates only the sector-detail-container div for smooth animation
    
    # Build candlestick chart card
    # Note: dcc.Graph objects don't evaluate to True in boolean context, so check explicitly
    if index_candlestick_chart is not None:
        candlestick_card = dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div([
                        html.Span("EMA 20", style={
                            "padding": "4px 12px",
                            "borderRadius": "15px",
                            "background": "rgba(255, 184, 0, 0.2)",
                            "border": "1px solid #FFB800",
                            "color": "#FFB800",
                            "fontSize": "0.75rem",
                            "fontWeight": "600",
                            "marginRight": "10px"
                        }),
                        html.Span(f"{INDEX_CANDLE_DAYS} Days", style={
                            "padding": "4px 12px",
                            "borderRadius": "15px",
                            "background": "rgba(0, 212, 255, 0.2)",
                            "border": "1px solid #00D4FF",
                            "color": "#00D4FF",
                            "fontSize": "0.75rem",
                            "fontWeight": "600"
                        })
                    ], style={"display": "flex", "alignItems": "center"})
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
            ], style={
                "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                "borderBottom": "2px solid #FFB800",
                "padding": "15px 20px"
            }),
            dbc.CardBody([
                index_candlestick_chart
            ], style={"padding": "15px"})
        ], style={
            "background": "linear-gradient(135deg, #12121c 0%, #1a1a2e 50%, #0f0f1a 100%)",
            "border": "1px solid #333",
            "borderRadius": "15px",
            "marginBottom": "20px",
            "boxShadow": "0 8px 32px rgba(255, 184, 0, 0.1), inset 0 1px 0 rgba(255,255,255,0.05)",
            "backdropFilter": "blur(10px)"
        })
    else:
        candlestick_card = None
    
    if vol_indicator:
        # Format the percentage change with proper sign
        vol_pct = vol_indicator["volume_pct_change"]
        vol_pct_display = f"+{vol_pct:.1f}%" if vol_pct >= 0 else f"{vol_pct:.1f}%"
        
        # Scaled volume (intraday projection)
        scaled_pct = vol_indicator["scaled_volume_pct_change"]
        scaled_display = f"+{scaled_pct:.1f}%" if scaled_pct >= 0 else f"{scaled_pct:.1f}%"
        
        # Market cap weighted
        mcap_weighted = vol_indicator["mcap_weighted_vol_change"]
        mcap_display = f"+{mcap_weighted:.1f}%" if mcap_weighted and mcap_weighted >= 0 else (f"{mcap_weighted:.1f}%" if mcap_weighted else "-")
        
        # Build the indicator card
        
        # Calculate breadth percentage
        breadth_pct = (vol_indicator['stocks_above_avg_vol'] / vol_indicator['valid_stocks'] * 100) if vol_indicator['valid_stocks'] > 0 else 0
        
        # Determine breadth color based on percentage
        breadth_color = "#00cc66" if breadth_pct >= 50 else "#FFB800" if breadth_pct >= 30 else "#ff4d4d"
        
        volume_indicator_card = dbc.Card([
            # Header with animated alert badge
            dbc.CardHeader([
                html.Div([
                    # Animated alert badge
                    html.Div(
                        vol_indicator["alert_text"],
                        className="pulse-animation" if vol_indicator["alert_level"] in ["extreme_high", "very_high"] else "",
                        style={
                            "padding": "8px 20px",
                            "borderRadius": "25px",
                            "background": f"linear-gradient(135deg, {vol_indicator['alert_bg']} 0%, {vol_indicator['alert_color']} 100%)",
                            "color": "#fff",
                            "fontSize": "0.85rem",
                            "fontWeight": "700",
                            "boxShadow": f"0 0 15px {vol_indicator['alert_bg']}60",
                            "border": f"1px solid {vol_indicator['alert_color']}40"
                        }
                    )
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
            ], style={
                "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                "borderBottom": "2px solid #00D4FF",
                "padding": "15px 20px"
            }),
            
            dbc.CardBody([
                # Main metrics row
                dbc.Row([
                    # Volume % Change (Main metric) - Larger and more prominent
                    dbc.Col([
                        html.Div([
                            html.Span("📈", style={"fontSize": "1rem", "marginRight": "8px"}),
                            html.Span("VOLUME % CHANGE", style={"color": "#888", "fontSize": "0.7rem", "letterSpacing": "1px"})
                        ], style={"marginBottom": "8px"}),
                        html.Div(
                            vol_pct_display,
                            className="big-number glow-animation" if abs(vol_pct) > 30 else "big-number",
                            style={
                                "fontSize": "3rem",
                                "fontWeight": "800",
                                "color": vol_indicator["alert_color"],
                                "lineHeight": "1",
                                "textShadow": f"0 0 30px {vol_indicator['alert_color']}40"
                            }
                        ),
                        html.Div(
                            f"vs {VOL_AVERAGE_DAYS}-day average",
                            style={"color": "#555", "fontSize": "0.7rem", "marginTop": "8px", "fontStyle": "italic"}
                        )
                    ], width=3, style={
                        "borderRight": "1px solid #333",
                        "paddingRight": "25px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center"
                    }),
                    
                    # Turnover Today - with icon
                    dbc.Col([
                        html.Div([
                            html.Span("💰", style={"fontSize": "1rem", "marginRight": "8px"}),
                            html.Span("TURNOVER TODAY", style={"color": "#888", "fontSize": "0.7rem", "letterSpacing": "1px"})
                        ], style={"marginBottom": "8px"}),
                        html.Div(
                            format_turnover_crores(vol_indicator["total_turnover_today"]),
                            style={
                                "fontSize": "1.6rem",
                                "fontWeight": "700",
                                "color": "#00D4FF",
                                "textShadow": "0 0 10px rgba(0, 212, 255, 0.3)"
                            }
                        ),
                        html.Div([
                            html.Span("Avg: ", style={"color": "#555"}),
                            html.Span(format_turnover_crores(vol_indicator['total_turnover_avg']), style={"color": "#777"})
                        ], style={"fontSize": "0.75rem", "marginTop": "5px"})
                    ], width=2, style={"borderRight": "1px solid #333", "padding": "0 15px"}),
                    
                    # Market Cap Weighted - with comparison indicator
                    dbc.Col([
                        html.Div([
                            html.Span("🏛️", style={"fontSize": "1rem", "marginRight": "8px"}),
                            html.Span("MCAP WEIGHTED", style={"color": "#888", "fontSize": "0.7rem", "letterSpacing": "1px"})
                        ], style={"marginBottom": "8px"}),
                        html.Div(
                            mcap_display,
                            style={
                                "fontSize": "1.6rem",
                                "fontWeight": "700",
                                "color": "#00cc66" if mcap_weighted and mcap_weighted >= 0 else "#ff4d4d"
                            }
                        ),
                        html.Div([
                            html.Span("Large caps " + ("leading" if mcap_weighted and mcap_weighted > vol_pct else "lagging"), 
                                     style={"color": "#00cc66" if mcap_weighted and mcap_weighted > vol_pct else "#ff4d4d", "fontSize": "0.7rem"})
                        ], style={"marginTop": "5px"})
                    ], width=2, style={"borderRight": "1px solid #333", "padding": "0 15px"}),
                    
                    # Volume Breadth - enhanced progress bar
                    dbc.Col([
                        html.Div([
                            html.Span("📊", style={"fontSize": "1rem", "marginRight": "8px"}),
                            html.Span("VOLUME BREADTH", style={"color": "#888", "fontSize": "0.7rem", "letterSpacing": "1px"})
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span(
                                f"{breadth_pct:.0f}%",
                                style={"color": breadth_color, "fontWeight": "700", "fontSize": "1.6rem"}
                            ),
                            html.Span(
                                f" ({vol_indicator['stocks_above_avg_vol']}/{vol_indicator['valid_stocks']})",
                                style={"color": "#666", "fontSize": "0.85rem", "marginLeft": "5px"}
                            ),
                        ]),
                        # Enhanced progress bar with segments
                        html.Div([
                            html.Div(
                                style={
                                    "width": f"{breadth_pct}%",
                                    "height": "8px",
                                    "background": f"linear-gradient(90deg, {breadth_color} 0%, {breadth_color}88 100%)",
                                    "borderRadius": "4px",
                                    "boxShadow": f"0 0 10px {breadth_color}50",
                                    "transition": "width 0.5s ease"
                                }
                            ),
                        ], style={
                            "width": "100%",
                            "height": "8px",
                            "background": "#2a2a2a",
                            "borderRadius": "4px",
                            "marginTop": "10px",
                            "overflow": "hidden"
                        }),
                        html.Div("above avg volume", style={"color": "#555", "fontSize": "0.65rem", "marginTop": "5px"})
                    ], width=2, style={"borderRight": "1px solid #333", "padding": "0 15px"}),
                    
                    # Flow Direction - enhanced with visual indicator
                    dbc.Col([
                        html.Div([
                            html.Span("🔄", style={"fontSize": "1rem", "marginRight": "8px"}),
                            html.Span("FLOW DIRECTION", style={"color": "#888", "fontSize": "0.7rem", "letterSpacing": "1px"})
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span(
                                "● " if vol_indicator["flow_color"] == "#00cc66" else "● ",
                                style={"color": vol_indicator["flow_color"], "fontSize": "1.2rem"}
                            ),
                            html.Span(
                                vol_indicator["flow_text"],
                                style={
                                    "fontSize": "1.1rem",
                                    "fontWeight": "700",
                                    "color": vol_indicator["flow_color"]
                                }
                            )
                        ]),
                        # Up/Down/Neutral counts with mini bars
                        html.Div([
                            html.Div([
                                html.Span(f"↑ {vol_indicator['stocks_up']}", style={"color": "#00cc66", "fontWeight": "600", "fontSize": "0.85rem"}),
                                html.Div(style={
                                    "width": f"{(vol_indicator['stocks_up'] / vol_indicator['valid_stocks'] * 100) if vol_indicator['valid_stocks'] > 0 else 0}%",
                                    "height": "3px",
                                    "background": "#00cc66",
                                    "borderRadius": "2px",
                                    "marginTop": "2px"
                                })
                            ], style={"flex": "1", "marginRight": "10px"}),
                            html.Div([
                                html.Span(f"↓ {vol_indicator['stocks_down']}", style={"color": "#ff4d4d", "fontWeight": "600", "fontSize": "0.85rem"}),
                                html.Div(style={
                                    "width": f"{(vol_indicator['stocks_down'] / vol_indicator['valid_stocks'] * 100) if vol_indicator['valid_stocks'] > 0 else 0}%",
                                    "height": "3px",
                                    "background": "#ff4d4d",
                                    "borderRadius": "2px",
                                    "marginTop": "2px"
                                })
                            ], style={"flex": "1"}),
                        ], style={"display": "flex", "marginTop": "8px"})
                    ], width=3, style={"padding": "0 15px"}),
                ], className="g-0"),
                
                # Intraday projection note (if market is open)
                html.Div([
                    html.Hr(style={"borderColor": "#333", "margin": "18px 0 12px 0"}),
                    html.Div([
                        html.Span("⏱️ ", style={"marginRight": "5px", "fontSize": "1rem"}),
                        html.Span("Intraday: ", style={"color": "#666", "fontWeight": "600"}),
                        html.Span(f"{vol_indicator['intraday_factor']*100:.0f}% of trading day", style={"color": "#888"}),
                        html.Span(" │ ", style={"color": "#333", "margin": "0 15px"}),
                        html.Span("Projected: ", style={"color": "#666", "fontWeight": "600"}),
                        html.Span(scaled_display, style={"color": "#00D4FF", "fontWeight": "700", "fontSize": "1.1rem"})
                    ], style={"fontSize": "0.85rem", "display": "flex", "alignItems": "center", "justifyContent": "center"})
                ]) if vol_indicator["intraday_factor"] < 1.0 else None,
                
                # Weekly Turnover Bar Chart
                html.Div([
                    html.Hr(style={"borderColor": "#333", "margin": "18px 0 12px 0"}),
                    html.Div([
                        html.Span("📊", style={"fontSize": "1rem", "marginRight": "8px"}),
                        html.Span("WEEKLY TURNOVER TREND", style={"color": "#888", "fontSize": "0.7rem", "letterSpacing": "1px", "fontWeight": "600"})
                    ], style={"marginBottom": "10px"}),
                    weekly_turnover_chart if weekly_turnover_chart is not None else html.Div("No weekly data available", style={"color": "#555", "fontSize": "0.8rem"})
                ])
                
            ], style={"padding": "25px"})
        ], style={
            "background": "linear-gradient(135deg, #12121c 0%, #1a1a2e 50%, #0f0f1a 100%)",
            "border": "1px solid #333",
            "borderRadius": "15px",
            "marginBottom": "20px",
            "boxShadow": "0 8px 32px rgba(0, 212, 255, 0.15), inset 0 1px 0 rgba(255,255,255,0.05)",
            "backdropFilter": "blur(10px)"
        })
    else:
        volume_indicator_card = html.Div(
            "Volume indicator not available - insufficient data",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "15px"}
        )
    # Build final layout with Accordion for indicator sections
    accordion_items = []
    
    # Accordion item for INDEX PRICE ACTION
    if candlestick_card is not None:
        accordion_items.append(
            dbc.AccordionItem(
                candlestick_card,
                title="📈 INDEX PRICE ACTION",
                item_id="candlestick-accordion",
            )
        )
    
    # Accordion item for SECTOR ROTATION TRACKER
    if sector_rotation_panel is not None:
        accordion_items.append(
            dbc.AccordionItem(
                sector_rotation_panel,
                title="🔄 SECTOR ROTATION",
                item_id="sector-accordion",
            )
        )
    
    # Accordion item for AGGREGATE VOLUME INDICATOR
    accordion_items.append(
        dbc.AccordionItem(
            volume_indicator_card,
            title="📊 AGGREGATE VOLUME INDICATOR",
            item_id="volume-accordion",
        )
    )
    
    # Create accordion with all sections open by default
    indicators_accordion = dbc.Accordion(
        accordion_items,
        active_item=["candlestick-accordion", "sector-accordion", "volume-accordion"],  # All open by default
        always_open=True,  # Allow multiple sections open at once
        flush=True,
        style={
            "marginBottom": "20px",
            "--bs-accordion-bg": "transparent",
            "--bs-accordion-btn-bg": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
            "--bs-accordion-active-bg": "transparent",
            "--bs-accordion-btn-color": "#fff",
            "--bs-accordion-btn-focus-box-shadow": "none",
            "--bs-accordion-border-color": "#333",
        }
    )
    
    # Build the final layout - sector detail panel is now inside the sector rotation card
    layout_items = [indicators_accordion, count_indicator, table]
    
    return html.Div(layout_items)


# -------------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Preload all indices on startup for instant switching
    preload_all_indices()
    app.run(debug=False, port=8051)