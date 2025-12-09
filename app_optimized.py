import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
from functools import lru_cache
from datetime import datetime
from nsepython import nse_eq
import time
from dash.dependencies import ALL

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
# 1) Load pre-generated symbol-industry mapping (FAST!)
# -------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_symbols_with_industries():
    """Load symbols and industries from pre-generated CSV - INSTANT LOAD!"""
    try:
        # Load from the pre-generated CSV with industries
        df = pd.read_csv("nifty100_with_industries.csv")
        
        print(f"✓ Loaded {len(df)} symbols with industries from CSV")
        
        # Create symbol -> industry mapping
        symbol_industry_map = dict(zip(df["symbol"], df["industry"]))
        
        # Remove N/A entries if you want
        # symbol_industry_map = {k: v for k, v in symbol_industry_map.items() if v != "N/A"}
        
        print(f"✓ Industry mapping ready with {len(symbol_industry_map)} stocks!")
        
        return symbol_industry_map
        
    except FileNotFoundError:
        print("ERROR: nifty100_with_industries.csv not found!")
        print("Please run fetch_static_data.py first to generate the file.")
        return {}
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}


SYMBOL_INDUSTRY_MAP = {}


# -------------------------------------------------------------------
# 2) CONSOLIDATED NSE DATA FETCH
# -------------------------------------------------------------------
@lru_cache(maxsize=256)
def get_nse_data(symbol: str):
    """Fetch full NSE data once and cache."""
    try:
        return nse_eq(symbol)
    except Exception as e:
        print(f"NSE Data Fetch Error for {symbol}: {e}")
        return None


# -------------------------------------------------------------------
# 3) FUNDAMENTALS for single stock
# -------------------------------------------------------------------
@lru_cache(maxsize=256)
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
@lru_cache(maxsize=256)
def get_historical_comparison(symbol: str, days: int):
    """Get price comparison for N days ago."""
    try:
        tk = yf.Ticker(symbol + ".NS")
        
        # Fetch historical data - get extra days to account for weekends/holidays
        hist = tk.history(period=f"{days + 10}d", interval="1d")
        
        if hist.empty or len(hist) < 2:
            return None, None, None
        
        # Get current price (most recent)
        current_price = float(hist['Close'].iloc[-1])
        
        # Get price from N days ago (or closest available)
        if len(hist) >= days:
            old_price = float(hist['Close'].iloc[-days])
        else:
            # Use oldest available if not enough data
            old_price = float(hist['Close'].iloc[0])
        
        # Calculate change
        price_change = current_price - old_price
        price_change_pct = (price_change / old_price * 100) if old_price != 0 else None
        
        return old_price, price_change, price_change_pct
        
    except Exception as e:
        print(f"Historical data error for {symbol}: {e}")
        return None, None, None


# -------------------------------------------------------------------
# 5) VOLUME STATS
# -------------------------------------------------------------------
@lru_cache(maxsize=256)
def get_volume_stats(symbol: str):
    """Return volume metrics with retry backoff for rate limits."""
    
    def _fetch_volume(sym):
        tk = yf.Ticker(sym + ".NS")
        
        avg_vol = None
        try:
            hist_30 = tk.history(period="30d", interval="1d")
            if "Volume" in hist_30.columns and not hist_30["Volume"].empty:
                avg_vol = float(hist_30["Volume"].mean())
        except Exception as e:
            print(f"  Avg volume fetch error for {sym}: {e}")

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
        }
    
    return retry_with_backoff(_fetch_volume, symbol, max_retries=3, base_delay=0.5) or {
        "avg_volume": None,
        "todays_volume": None,
        "volume_change_pct": None,
    }


# -------------------------------------------------------------------
# 6) AGGREGATE VOLUME INDICATOR (Market-Wide)
# -------------------------------------------------------------------
from datetime import time as dt_time

def calculate_aggregate_volume_indicator(stocks_data):
    """
    Calculate aggregate volume % change indicator for all stocks.
    
    Uses price-weighted volume (turnover) for proper normalization.
    This accounts for different stock prices when aggregating volume.
    
    Returns:
        dict with volume metrics, alert level, and market breadth
    """
    if not stocks_data:
        return None
    
    # Initialize accumulators
    total_turnover_today = 0          # Price × Today's Volume
    total_turnover_avg = 0            # Price × Average Volume
    
    # Directional flow tracking
    up_turnover = 0                   # Turnover from stocks going up
    down_turnover = 0                 # Turnover from stocks going down
    
    # Market breadth counters
    stocks_above_avg_vol = 0          # Stocks with above average volume
    stocks_below_avg_vol = 0          # Stocks with below average volume
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
        
        # Skip if essential data is missing
        if current_price is None or today_volume is None or avg_volume is None:
            continue
        if current_price <= 0 or avg_volume <= 0:
            continue
        
        valid_stocks += 1
        
        # Calculate turnover (Price × Volume)
        turnover_today = current_price * today_volume
        turnover_avg = current_price * avg_volume
        
        total_turnover_today += turnover_today
        total_turnover_avg += turnover_avg
        
        # Track individual volume changes for stats
        if vol_change_pct is not None:
            individual_vol_changes.append(vol_change_pct)
        
        # Market breadth: above/below average volume
        if today_volume > avg_volume:
            stocks_above_avg_vol += 1
        else:
            stocks_below_avg_vol += 1
        
        # Market cap weighted volume change
        if market_cap is not None and vol_change_pct is not None:
            total_market_cap += market_cap
            weighted_vol_change += market_cap * vol_change_pct
        
        # Directional flow (up vs down stocks)
        if price_change_pct is not None:
            if price_change_pct > 0:
                up_turnover += turnover_today
                stocks_up += 1
            elif price_change_pct < 0:
                down_turnover += turnover_today
                stocks_down += 1
            else:
                stocks_neutral += 1
        else:
            stocks_neutral += 1
    
    # Avoid division by zero
    if total_turnover_avg == 0 or valid_stocks == 0:
        return None
    
    # ========== CORE CALCULATIONS ==========
    
    # 1. Volume Ratio (Today vs Average)
    volume_ratio = total_turnover_today / total_turnover_avg
    volume_pct_change = (volume_ratio - 1) * 100
    
    # 2. Market Cap Weighted Volume Change
    mcap_weighted_vol_change = None
    if total_market_cap > 0:
        mcap_weighted_vol_change = weighted_vol_change / total_market_cap
    
    # 3. Net Money Flow
    net_flow = up_turnover - down_turnover
    total_flow = up_turnover + down_turnover
    flow_ratio = (up_turnover / total_flow * 100) if total_flow > 0 else 50
    
    # 4. Intraday Volume Scaling (if market is open)
    # Indian market hours: 9:15 AM to 3:30 PM IST
    current_time = datetime.now()
    market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    intraday_factor = 1.0
    scaled_volume_pct_change = volume_pct_change
    
    if market_open <= current_time <= market_close:
        # Calculate how much of the trading day has passed
        total_trading_seconds = (market_close - market_open).total_seconds()
        elapsed_seconds = (current_time - market_open).total_seconds()
        day_progress = elapsed_seconds / total_trading_seconds
        
        if day_progress > 0:
            # Scale expected volume based on time elapsed
            intraday_factor = day_progress
            # Projected full-day turnover
            projected_turnover = total_turnover_today / day_progress
            scaled_volume_ratio = projected_turnover / total_turnover_avg
            scaled_volume_pct_change = (scaled_volume_ratio - 1) * 100
    
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
# 7) FETCH DATA FOR SYMBOLS IN SELECTED INDUSTRY
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
    suppress_callback_exceptions=True
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
    
    # Enhanced Header
    html.Div([
        html.H1([
            html.Span("📊 ", style={"marginRight": "10px"}),
            html.Span("AKS Market", style={
                "background": "linear-gradient(135deg, #00D4FF 0%, #00FF88 50%, #00D4FF 100%)",
                "backgroundClip": "text",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "fontWeight": "800",
            }),
            html.Span(" - NIFTY100", style={"color": "#888", "fontWeight": "400", "fontSize": "0.7em"})
        ], style={"margin": "0", "fontSize": "2.2rem", "letterSpacing": "-1px"}),
        html.P("Real-time stock screener with aggregate volume analytics", 
               style={"color": "#666", "fontSize": "0.9rem", "margin": "5px 0 0 0", "letterSpacing": "0.5px"})
    ], className="text-center", style={"padding": "20px 0 25px 0"}),
    
    dbc.Row([
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
            html.Div(id="update-timestamp", style={"textAlign": "right", "color": "#00D4FF", "fontSize": "0.9rem", "fontWeight": "600", "marginTop": "35px"})
        ], width=5),
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
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-1",
                type="circle",
                color="#00D4FF",
                children=html.Div(id="table-container", style={"overflowX": "auto", "minHeight": "200px"})
            )
        ], width=12),
    ])
    
], fluid=True, style={"backgroundColor": "#1a1a1a", "color": "#fff", "padding": "20px"})


# -------------------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------------------

# Callback 1: Load initial symbol-industry mapping (INSTANT NOW!)
@app.callback(
    [Output("symbol-industry-map", "data"),
     Output("industry-filter", "options")],
    Input("industry-filter", "id")
)
def initialize_data(_):
    """Load symbol-industry mapping from CSV - INSTANT LOAD!"""
    global SYMBOL_INDUSTRY_MAP
    
    print("Loading pre-generated industry data...")
    SYMBOL_INDUSTRY_MAP = load_symbols_with_industries()
    
    if not SYMBOL_INDUSTRY_MAP:
        print("WARNING: No data loaded! Please run fetch_static_data.py first!")
        return {}, []
    
    industries = set(SYMBOL_INDUSTRY_MAP.values())
    industries.discard("N/A")
    
    # Add "All" option at the beginning
    options = [{"label": "🌐 All Industries", "value": "ALL"}]
    options.extend([{"label": ind, "value": ind} for ind in sorted(industries)])
    
    print(f"✓ Dropdown ready with {len(options)} options (including 'All')")
    
    return SYMBOL_INDUSTRY_MAP, options


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
        get_nse_data.cache_clear()
        get_fundamentals.cache_clear()
        get_volume_stats.cache_clear()
        get_historical_comparison.cache_clear()
        print(f"[{trigger_source.upper()}] Caches cleared at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get symbols for selected industry or all symbols
    if selected_industry == "ALL":
        symbols_in_industry = list(symbol_industry_map.keys())
        display_name = "All Industries"
    else:
        symbols_in_industry = [symbol for symbol, industry in symbol_industry_map.items() 
                              if industry == selected_industry]
        display_name = selected_industry
    
    if not symbols_in_industry:
        return {}, f"No stocks found for {display_name}", days_comparison
    
    print(f"\n[FETCHING DATA - {trigger_source}] Loading {len(symbols_in_industry)} stocks for: {display_name} (comparing {days_comparison} days)")
    
    # Fetch data with historical comparison
    stocks_data = fetch_stocks_data_for_industry(symbols_in_industry, days_comparison=days_comparison)
    
    # Create timestamp with source indicator
    now = datetime.now().strftime("%H:%M:%S")
    timestamp = f"Last updated: {now} | {len(stocks_data)} stocks | {days_comparison}D comparison | Next refresh: 5 min"
    
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
    triggered_id = eval(triggered)   # Convert string dict to actual dict

    column = triggered_id["column"]

    # Toggle direction
    if column == current_column:
        new_direction = "desc" if current_direction == "asc" else "asc"
    else:
        new_direction = "desc"  # default

    return column, new_direction

# Callback 5: Generate table with sorting
@app.callback(
    Output("table-container", "children"),
    [Input("stocks-data-store", "data"),
     Input("industry-filter", "value"),
     Input("current-days", "data"),
     Input("sort-column", "data"),
     Input("sort-direction", "data")]
)
def generate_table(stocks_data_store, selected_industry, days, sort_column, sort_direction):
    """Generate table with stock data and sorting."""

    if not selected_industry:
        return html.Div([
            # Welcome Section
            html.Div([
                html.Div([
                    html.Span("🚀", style={"fontSize": "4rem", "marginBottom": "20px", "display": "block"}),
                    html.H2("Welcome to AKS Market Screener", style={
                        "color": "#fff",
                        "fontWeight": "700",
                        "fontSize": "2rem",
                        "marginBottom": "10px"
                    }),
                    html.P("Your real-time stock analysis dashboard with aggregate volume insights", style={
                        "color": "#888",
                        "fontSize": "1rem",
                        "marginBottom": "30px"
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
                        html.Span(" to see the complete Nifty 100 aggregate volume indicator", style={"color": "#888"})
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


    # Header
    header_row = html.Tr([
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
        
        row = html.Tr([
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
                    html.Div([
                        html.Span("📊", style={"fontSize": "1.5rem", "marginRight": "12px"}),
                        html.Span("AGGREGATE VOLUME INDICATOR", style={
                            "fontSize": "1rem",
                            "fontWeight": "700",
                            "letterSpacing": "1px",
                            "background": "linear-gradient(90deg, #fff 0%, #00D4FF 100%)",
                            "backgroundClip": "text",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                        }),
                    ], style={"display": "flex", "alignItems": "center"}),
                    
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
                            "vs 30-day average",
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
                ]) if vol_indicator["intraday_factor"] < 1.0 else None
                
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
    
    return html.Div([volume_indicator_card, count_indicator, table])


# -------------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=8051)