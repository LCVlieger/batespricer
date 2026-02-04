import os
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
from scipy.interpolate import PchipInterpolator, interp1d
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 
    bid: float = 0.0
    ask: float = 0.0

# --- PART 1: RATE ENGINE ---
class NSSYieldCurve:
    def __init__(self, curve_fit):
        self.curve = curve_fit
    def get_rate(self, T: float) -> float:
        return float(self.curve(max(T, 1e-4)))
    def to_dict(self):
        # Sample for meta storage
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
            return NSSYieldCurve(curve_fit)
    raise ValueError("Could not fetch FRED rates.")
class ImpliedDividendCurve:
    """
    Hybrid dividend engine. Uses hardcoded values for front-end pillars 
    (1W, 2W) to eliminate calibration needles, falling back to 
    ATM-synced Put-Call Parity for longer tenors.
    """
    def __init__(self, df: pd.DataFrame, S0_anchor: float, r_curve):
        self.yields = {}
        
        # --- HARDCODED PILLARS ---
        # These values target the 'sweet spots' that eliminate 
        # pricing needles for short-dated ITM/OTM wings.
        # Format: {Maturity_Years: Yield_Value}
        manual_pins = {
            0.019: 0.13, #-0.055, #-0.055,  # 1-Week: Pinning at -5.5%
            0.041: 0.13 # -0.03 #-0.030   # 2-Weeks: Pinning at -3.0%
        }
        
        unique_Ts = sorted([t for t in df['T'].unique()]) 
        
        for T in unique_Ts:
            # Check for a manual override (with 1-day tolerance)
            override_val = next((v for m, v in manual_pins.items() if abs(m - T) < 0.005), None)
            
            if override_val is not None:
                self.yields[T] = override_val
                continue

            # --- FALLBACK: ATM SYNC ---
            subset = df[df['T'] == T]
            # Find the strike closest to spot
            atm_idx = (subset['STRIKE'] - S0_anchor).abs().idxmin()
            atm_opt = subset.loc[atm_idx]
            
            # Put-Call Parity: F = (C - P) * e^(rT) + K
            r = r_curve.get_rate(T)
            C, P, K = atm_opt['C_MID'], atm_opt['P_MID'], atm_opt['STRIKE']
            
            # Back-solve for Market-Implied Forward (F)
            F_market = (C - P) * np.exp(r * T) + K
            
            # F = S0 * e^((r-q)T)  =>  q = r - ln(F/S0) / T
            if T > 1e-4:
                q_sync = r - np.log(F_market / S0_anchor) / T
            else:
                q_sync = 0.0
            
            self.yields[T] = np.clip(q_sync, -0.20, 0.20)

        # --- INTERPOLATION ---
        mats = np.array(sorted(self.yields.keys()))
        vals = np.array([self.yields[m] for m in mats])
        
        # Linear interpolation handles the jagged front-end better than cubic/Pchip
        self.interpolator = interp1d(mats, vals, kind='linear', 
                                     bounds_error=False, 
                                     fill_value=(vals[0], vals[-1]))
        
        self.min_T, self.max_T = mats[0], mats[-1]
        self.val_min, self.val_max = vals[0], vals[-1]

    def get_rate(self, T: float) -> float:
        if T < self.min_T: return float(self.val_min)
        if T > self.max_T: return float(self.val_max)
        return float(self.interpolator(T))

    def to_dict(self):
        return {str(round(k, 3)): v for k, v in self.yields.items()}
    
# --- PART 3: DATA FETCHING & DYNAMIC SPOT ANCHOR ---
def fetch_raw_data(ticker_symbol: str) -> pd.DataFrame:
    """Parallel fetch of liquid monthly/quarterly contracts."""
    ticker = yf.Ticker(ticker_symbol)
    today = datetime.now()
    targets = [0.019, 0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.25] 
    
    selected_exps = set()
    all_exps = ticker.options
    for t in targets:
        best = min(all_exps, key=lambda x: abs(((datetime.strptime(x, "%Y-%m-%d") - today).days / 365.25) - t))
        selected_exps.add(best)

    def fetch_one(exp_str):
        try:
            T = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days / 365.25
            chain = ticker.option_chain(exp_str)
            df_c = chain.calls[['strike', 'bid', 'ask']].rename(columns={'strike':'STRIKE', 'bid':'bC', 'ask':'aC'})
            df_p = chain.puts[['strike', 'bid', 'ask']].rename(columns={'strike':'STRIKE', 'bid':'bP', 'ask':'aP'})
            full = df_c.merge(df_p, on='STRIKE')
            full['C_MID'], full['P_MID'], full['T'] = (full['bC']+full['aC'])/2, (full['bP']+full['aP'])/2, T
            return full
        except: return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch_one, sorted(list(selected_exps))))
    return pd.concat([r for r in results if r is not None], ignore_index=True)

def get_market_implied_spot(ticker_symbol: str, raw_df: pd.DataFrame, r_curve) -> float:
    """
    DERIVES THE SPOT ANCHOR FROM MARKET DATA.
    Uses long-tenor stability to estimate the dividend baseline, then
    synchronizes the 1-month forward to find the true spot.
    """
    ticker = yf.Ticker(ticker_symbol)
    S_cash = ticker.fast_info['last_price']
    
    # 1. Detect the current Market Baseline Yield from stable long tenors (T > 0.5)
    stable_Ts = sorted([t for t in raw_df['T'].unique() if t >= 0.5])
    baseline_qs = []
    
    for T in stable_Ts:
        subset = raw_df[raw_df['T'] == T]
        X = subset['STRIKE'].values.reshape(-1, 1)
        y = (subset['C_MID'] - subset['P_MID']).values
        reg = LinearRegression().fit(X, y)
        F_T = -reg.intercept_ / reg.coef_[0]
        r_T = r_curve.get_rate(T)
        
        # Estimate raw q relative to cash (low noise at T > 0.5)
        baseline_qs.append(r_T - np.log(F_T / S_cash) / T)
    
    market_baseline_q = np.median(baseline_qs) if baseline_qs else 0.013
    print(f"Detected Market Baseline q: {market_baseline_q:.4%}")
    
    # 2. Use the 1-month pillar to find the synchronized Spot
    anchor_T = min(raw_df['T'].unique(), key=lambda x: abs(x - 0.0833))
    subset_anchor = raw_df[raw_df['T'] == anchor_T]
    
    X_a = subset_anchor['STRIKE'].values.reshape(-1, 1)
    y_a = (subset_anchor['C_MID'] - subset_anchor['P_MID']).values
    reg_a = LinearRegression().fit(X_a, y_a)
    F_anchor = -reg_a.intercept_ / reg_a.coef_[0]
    r_anchor = r_curve.get_rate(anchor_T)
    
    # Synchronized S0 = Forward / exp((r - baseline_q) * T)
    # This aligns the Spot to the options plane using the baseline yield.
    S0_synchronized = F_anchor / np.exp((r_anchor - market_baseline_q) * anchor_T)
    
    return float(S0_synchronized)

def fetch_options(ticker_symbol: str, S0: float, target_size: int = 300) -> List[MarketOption]:
    ticker = yf.Ticker(ticker_symbol)
    today = datetime.now()
    # Filter for standard liquid maturities (>= 2 weeks)
    valid_exps = [e for e in ticker.options if 0.04 <= (datetime.strptime(e, "%Y-%m-%d") - today).days/365.25 <= 1.3]
    
    def process(exp_str):
        try:
            T = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days / 365.25
            chain = ticker.option_chain(exp_str)
            local = []
            for opt_type, data, f in [('PUT', chain.puts, lambda k: k < S0), ('CALL', chain.calls, lambda k: k > S0)]:
                subset = data[f(data['strike']) & (data['strike'] > S0*0.75) & (data['strike'] < S0*1.25)]
                for _, row in subset.iterrows():
                    mid, bid, ask = (row['bid']+row['ask'])/2, row['bid'], row['ask']
                    if mid > 0.1 and bid > 0 and (ask-bid)/max(mid, 0.01) < 0.25:
                        local.append(MarketOption(row['strike'], T, mid, opt_type, bid, ask))
            return local
        except: return []

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process, valid_exps))
    
    all_c = [item for sublist in results for item in sublist]
    if len(all_c) > target_size:
        indices = np.linspace(0, len(all_c)-1, target_size, dtype=int)
        return [all_c[i] for i in indices]
    return all_c