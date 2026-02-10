import os
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass
from dataclasses import asdict
from typing import List, Dict
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION: THE BOX SPREAD ---
# SPX options trade on a rate higher than Treasuries (Box Spread/SOFR).
# Currently, this spread is approximately 40-50 basis points (0.0045).
BOX_SPREAD = 0.0045

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0 # Capture real spread for CSV/Weights

def save_options_to_cache(options: List[MarketOption], ticker: str):
    """Saves the list of MarketOption objects to a JSON file."""
    os.makedirs("cache", exist_ok=True)
    filename = f"cache/options_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
    
    # Convert dataclasses to dictionaries for JSON serialization
    data_to_save = [asdict(o) for o in options]
    
    with open(filename, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Successfully cached {len(options)} options to {filename}")
    return filename

def load_options_from_cache(filepath: str) -> List[MarketOption]:
    """Loads options from a previously saved JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return [MarketOption(**item) for item in data]

# --- PART 1: RATE ENGINE ---
class NSSYieldCurve:
    def __init__(self, curve_fit, spread=0.0):
        self.curve = curve_fit
        self.spread = spread

    def get_rate(self, T: float) -> float:
        """Returns the Treasury rate plus the Box Spread adjustment."""
        return float(self.curve(max(T, 1e-4))) + self.spread

    def to_dict(self):
        return {f"{round(t,3)}Y": self.get_rate(t) for t in [0.08, 0.25, 0.5, 1.0]}

def fetch_treasury_rates_fred(date_str: str, api_key: str) -> NSSYieldCurve:
    series_map = {1/12: "DGS1MO", 3/12: "DGS3MO", 6/12: "DGS6MO", 1.0: "DGS1", 2.0: "DGS2"}
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    for i in range(6):
        d_str = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        mats, yields = [], []
        for tenor, s_id in series_map.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json&observation_start={d_str}&observation_end={d_str}"
                res = requests.get(url, timeout=3).json()
                val = res['observations'][0]['value']
                if val != '.':
                    mats.append(tenor)
                    yields.append(float(val) / 100.0)
            except: continue
        if len(mats) >= 3:
            from nelson_siegel_svensson.calibrate import calibrate_nss_ols
            curve_fit, _ = calibrate_nss_ols(np.array(mats), np.array(yields))
            # Initialize with the global BOX_SPREAD adjustment
            return NSSYieldCurve(curve_fit, spread=BOX_SPREAD)
    raise ValueError("Could not fetch FRED rates.")

# --- PART 2: SETTLEMENT & TIME ENGINE ---

def is_third_friday(d: datetime) -> bool:
    """Detects if a date is the 3rd Friday of the month (Standard SPX Monthly)."""
    return d.weekday() == 4 and 15 <= d.day <= 21

def calculate_spx_time_to_maturity(expiry_date: datetime, ticker_symbol: str) -> float:
    """Standardizes T for SPX AM vs PM settlement."""
    now = datetime.now()
    
    # Check for AM Settled Indices (SPX, NDX, RUT, VIX)
    # Note: SPY, QQQ, and single stocks are PM settled.
    am_settled_tickers = ["^SPX", "^NDX", "^VIX", "^RUT", "^GDAXI"]
    is_am_settled = any(ticker_symbol.startswith(t) for t in am_settled_tickers)

    if is_am_settled and is_third_friday(expiry_date):
        # AM Settlement (9:30 AM ET)
        expiry_settlement = datetime.combine(expiry_date.date(), dt_time(9, 30))
    else:
        # PM Settlement (4:00 PM ET) - Stocks/ETFs
        expiry_settlement = datetime.combine(expiry_date.date(), dt_time(16, 0))
        
    delta = expiry_settlement - now
    # Ensure T is at least 1 minute to avoid divide-by-zero
    return max(delta.total_seconds() / (365.25 * 24 * 3600), 1e-6)

# --- PART 3: DIVIDEND ENGINE ---

class ImpliedDividendCurve:
    def __init__(self, df: pd.DataFrame, S0_anchor: float, r_curve, ticker_symbol: str = ""):
        self.yields = {}
        unique_Ts = sorted(df['T'].unique()) 
        
        # 1. Detect if this is an Index or a Stock
        # Indices (SPX, NDX) -> Use Market Implied Parity (Your original logic)
        # Stocks (NVDA, AAPL) -> Use Fundamental Yield (Fixes the 8.9% ghost yield)
        is_index = ticker_symbol.startswith("^") or ticker_symbol in ["SPX", "NDX", "RUT"]
        
        fundamental_q = 0.0
        if not is_index:
            try:
                # Fetch fundamental yield for American stocks
                t_obj = yf.Ticker(ticker_symbol)
                raw_q = t_obj.info.get('dividendYield')/100 # Returns e.g. 0.002
                fundamental_q = raw_q if raw_q is not None else 0.0
                
                # Sanity check: If Yahoo returns None or crazy data, default to 0
                if fundamental_q > 0.15: fundamental_q = 0.0 
            except:
                fundamental_q = 0.0

        for T in unique_Ts:
            if is_index:
                # === INDEX MODE (IMPLIED PARITY) ===
                subset = df[df['T'] == T]
                r = r_curve.get_rate(T)
                
                parity_values = (subset['C_MID'] - subset['P_MID']) + subset['STRIKE'] * np.exp(-r * T)
                S0_q_intercept = np.median(parity_values)
                
                if T > 1e-4 and S0_q_intercept > 0:
                    q_sync = -np.log(S0_q_intercept / S0_anchor) / T
                else:
                    q_sync = 0.0
                
                # Clip to realistic index bounds
                self.yields[T] = float(np.clip(q_sync, -0.01, 0.06))
            
            else:
                # === STOCK MODE (FUNDAMENTAL) ===
                # American options break parity calculations. 
                # Trusting the fundamental yield is safer and more accurate.
                self.yields[T] = fundamental_q

        # Interpolation Setup
        mats = np.array(sorted(self.yields.keys()))
        vals = np.array([self.yields[m] for m in mats])
        
        self.interpolator = interp1d(mats, vals, kind='linear', 
                                     bounds_error=False, 
                                     fill_value=(vals[0], vals[-1]))
        
        self.min_T, self.max_T = mats[0], mats[-1]

    def get_rate(self, T: float) -> float:
        if T < self.min_T: return float(self.yields[self.min_T])
        if T > self.max_T: return float(self.yields[self.max_T])
        return float(self.interpolator(T))

    def to_dict(self):
        return {str(round(k, 4)): v for k, v in self.yields.items()}
    
# --- PART 4: DATA FETCHING ---

def fetch_raw_data(ticker_symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(ticker_symbol)
    try:
        all_exps = ticker.options
    except:
        return pd.DataFrame()
        
    if not all_exps:
        return pd.DataFrame()

    targets = [0.019, 0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.25] 
    
    selected_exps = set()
    for t in targets:
        # FIXED: ticker_symbol is now correctly passed inside the function call
        best = min(all_exps, key=lambda x: abs(calculate_spx_time_to_maturity(datetime.strptime(x, "%Y-%m-%d"), ticker_symbol) - t))
        selected_exps.add(best)

    def fetch_one(exp_str):
        try:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            # FIXED: Pass ticker_symbol
            T = calculate_spx_time_to_maturity(exp_dt, ticker_symbol)
            chain = ticker.option_chain(exp_str)
            df_c = chain.calls[['strike', 'bid', 'ask']].rename(columns={'strike':'STRIKE', 'bid':'bC', 'ask':'aC'})
            df_p = chain.puts[['strike', 'bid', 'ask']].rename(columns={'strike':'STRIKE', 'bid':'bP', 'ask':'aP'})
            
            # Use Inner Join for Parity Checks
            full = df_c.merge(df_p, on='STRIKE', how='inner')
            if full.empty: return None
            
            full['C_MID'], full['P_MID'], full['T'] = (full['bC']+full['aC'])/2, (full['bP']+full['aP'])/2, T
            return full
        except: return None

    # Reduce workers for individual stocks to avoid API blocking
    workers = 4 if ticker_symbol.startswith("^") else 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_one, sorted(list(selected_exps))))
    
    clean_results = [r for r in results if r is not None and not r.empty]
    if not clean_results: return pd.DataFrame()
        
    return pd.concat(clean_results, ignore_index=True)

def get_market_implied_spot(ticker_symbol: str, raw_df: pd.DataFrame, r_curve) -> float:
    ticker = yf.Ticker(ticker_symbol)
    
    # === 1. STOCK LOGIC: Prioritize Cash Price ===
    # We use the updated 'fast_info' logic for anything without the '^' prefix.
    if not ticker_symbol.startswith("^"):
        try:
            fi = ticker.fast_info
            if 'last_price' in fi and fi['last_price'] > 0:
                return float(fi['last_price'])
            
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception: 
            pass # Fallback to parity if cash fetch fails

    # === 2. INDEX LOGIC: Your EXACT Original Regression Path ===
    # If it's an index (or stock fetch failed), use the linear regression logic.
    
    if raw_df is None or raw_df.empty:
        hist = ticker.history(period="1d")
        if not hist.empty: return float(hist['Close'].iloc[-1])
        raise ValueError(f"No data available for {ticker_symbol}")

    # Your original logic: Use the nearest expiration (approx 1 month) to anchor
    anchor_T = min(raw_df['T'].unique(), key=lambda x: abs(x - 0.0833))
    subset_anchor = raw_df[raw_df['T'] == anchor_T]
    
    if len(subset_anchor) > 5:
        # Linear Regression: (C - P) = e^(-rT) * S - e^(-rT) * K
        # Here we solve for the Forward price F = -intercept / slope
        X = subset_anchor['STRIKE'].values.reshape(-1, 1)
        y = (subset_anchor['C_MID'] - subset_anchor['P_MID']).values
        reg_a = LinearRegression().fit(X, y)
        
        if abs(reg_a.coef_[0]) > 1e-5:
            F_anchor = -reg_a.intercept_ / reg_a.coef_[0]
            r_anchor = r_curve.get_rate(anchor_T)
            
            # Use your original 1.3% baseline for indices vs 0 for others
            market_baseline_q = 0.013 if ticker_symbol.startswith("^") else 0.0
            
            S0_synchronized = F_anchor / np.exp((r_anchor - market_baseline_q) * anchor_T)
            
            print(f"[{ticker_symbol}] Synced Index Spot (Regression) | T={anchor_T:.3f} | S0: {S0_synchronized:.3f}")
            return float(S0_synchronized)

    # Absolute fallback to Yahoo History
    try:
        hist = ticker.history(period="1d")
        return float(hist['Close'].iloc[-1])
    except:
        return 100.0

def fetch_options(ticker_symbol: str, S0: float, target_size: int = 300) -> List[MarketOption]:
    if np.isnan(S0): return []

    ticker = yf.Ticker(ticker_symbol)
    try:
        all_exps = ticker.options
    except: return []

    valid_exps = []
    for e in all_exps:
        try:
            # FIXED: Pass ticker_symbol
            t_val = calculate_spx_time_to_maturity(datetime.strptime(e, "%Y-%m-%d"), ticker_symbol)
            if 0.04 <= t_val <= 1.3:
                valid_exps.append(e)
        except: continue
    
    def process(exp_str):
        try:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            # FIXED: Pass ticker_symbol
            T = calculate_spx_time_to_maturity(exp_dt, ticker_symbol)
            chain = ticker.option_chain(exp_str)
            local = []
            
            # === CRITICAL FOR AMERICAN OPTIONS: STRICT OTM FILTERING ===
            # We filter strictly for Out-of-the-Money (OTM) options.
            # ITM American options carry early exercise premiums that distort the curve.
            # OTM Calls (K > S0) and OTM Puts (K < S0) are "cleaner".
            
            # Using 1.02 and 0.98 buffers to avoid ATM noise
            for opt_type, data, f in [('PUT', chain.puts, lambda k: k < S0 * 0.98), 
                                      ('CALL', chain.calls, lambda k: k > S0 * 1.02)]:
                
                # Broad strike filter around spot (70% - 130%)
                subset = data[f(data['strike']) & (data['strike'] > S0*0.70) & (data['strike'] < S0*1.30)]
                
                for _, row in subset.iterrows():
                    mid, bid, ask = (row['bid']+row['ask'])/2, row['bid'], row['ask']
                    
                    # Spread & Liquidity Filters
                    if mid > 0.05 and bid > 0 and (ask-bid)/max(mid, 0.01) < 0.25:
                        local.append(MarketOption(row['strike'], T, mid, opt_type, bid, ask))
            return local
        except: return []

    # Workers = 4 for Index, 1 for Stock to be safe
    workers = 4 if ticker_symbol.startswith("^") else 1
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process, valid_exps))
    
    all_c = [item for sublist in results for item in sublist]
    if len(all_c) > target_size:
        indices = np.linspace(0, len(all_c)-1, target_size, dtype=int)
        return [all_c[i] for i in indices]
    return all_c