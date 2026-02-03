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
from scipy.interpolate import PchipInterpolator
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

# --- PART 2: ROBUST DIVIDEND ENGINE ---
class ImpliedDividendCurve:
    """
    Industry-grade dividend engine. Uses Regression PCP on liquid pillars.
    Ignores short-end noise (<1M) to prevent yield explosions.
    """
    def __init__(self, df: pd.DataFrame, S0_anchor: float, r_curve):
        self.yields = {}
        
        # Filter pillars: Only use 1M to 1.3Y (Stability over Noise)
        # We ignore T < 0.07 to avoid the 1/T noise magnification
        unique_Ts = sorted([t for t in df['T'].unique() if 0.07 <= t <= 1.3])
        
        for T in unique_Ts:
            subset = df[df['T'] == T]
            # Use ATM options (+/- 8% moneyness) for regression stability
            mask = (subset['STRIKE'] > S0_anchor * 0.92) & (subset['STRIKE'] < S0_anchor * 1.08)
            data = subset[mask].dropna()
            
            if len(data) < 5: continue
            
            # Regression PCP: (C - P) = exp(-rT) * F - exp(-rT) * K
            # Model: y = alpha + beta * K
            X = data['STRIKE'].values.reshape(-1, 1)
            y = (data['C_MID'] - data['P_MID']).values
            
            reg = LinearRegression().fit(X, y)
            alpha, beta = reg.intercept_, reg.coef_[0]
            
            # F = -alpha / beta (Market implied forward)
            if beta < 0:
                F_implied = -alpha / beta
                r = r_curve.get_rate(T)
                # Solve for q: F = S0 * exp((r-q)T) -> q = r - ln(F/S0)/T
                q = r - np.log(F_implied / S0_anchor) / T
                
                # Clamp to realistic SPX range (0.2% to 2.5%)
                self.yields[T] = max(0.002, min(q, 0.025))

        # Setup Monotonic Cubic Spline (Pchip)
        mats = np.array(sorted(self.yields.keys()))
        vals = np.array([self.yields[m] for m in mats])
        
        if len(mats) > 1:
            self.interpolator = PchipInterpolator(mats, vals, extrapolate=False)
            self.min_T, self.max_T = mats[0], mats[-1]
            self.val_min, self.val_max = vals[0], vals[-1]
        else:
            default = vals[0] if len(vals) > 0 else 0.013 # SPX default
            self.interpolator = lambda t: default
            self.min_T, self.max_T, self.val_min, self.val_max = 0, 99, default, default

    def get_rate(self, T: float) -> float:
        # Flat extrapolation outside the pillar range for stability
        if T < self.min_T: return float(self.val_min)
        if T > self.max_T: return float(self.val_max)
        return float(self.interpolator(T))
    
    def to_dict(self):
        return {str(round(k,3)): v for k,v in self.yields.items()}

# --- PART 3: OPTIMIZED DATA FETCHING ---
def fetch_raw_data(ticker_symbol: str) -> pd.DataFrame:
    """Parallel fetch of liquid monthly/quarterly monthly pillars."""
    ticker = yf.Ticker(ticker_symbol)
    today = datetime.now()
    # Focus on standard monthly tenors for a clean curve
    targets = [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.25] 
    
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
        results = list(executor.map(fetch_one, selected_exps))
    return pd.concat([r for r in results if r is not None], ignore_index=True)

def get_market_implied_spot(ticker_symbol: str, raw_df: pd.DataFrame, r_curve) -> float:
    """
    Derives a synchronized S0 anchor.
    This prevents run-to-run drift by aligning spot with the option prices.
    """
    # Use the 1-month pillar as the master anchor
    unique_Ts = sorted(raw_df['T'].unique())
    anchor_T = min(unique_Ts, key=lambda x: abs(x - 0.08))
    
    subset = raw_df[raw_df['T'] == anchor_T].copy()
    r = r_curve.get_rate(anchor_T)
    
    # Simple Regression to find the Forward
    X = subset['STRIKE'].values.reshape(-1, 1)
    y = (subset['C_MID'] - subset['P_MID']).values
    reg = LinearRegression().fit(X, y)
    F = -reg.intercept_ / reg.coef_[0]
    
    # We anchor S0 by assuming a baseline SPX yield of ~1.3% at 1-month
    S0_consistent = F * np.exp(-(r - 0.013) * anchor_T)
    return float(S0_consistent)

def fetch_options(ticker_symbol: str, S0: float, target_size: int = 300) -> List[MarketOption]:
    """Fetches the broad calibration dataset."""
    ticker = yf.Ticker(ticker_symbol)
    today = datetime.now()
    valid_exps = [e for e in ticker.options if 0.04 <= (datetime.strptime(e, "%Y-%m-%d") - today).days/365.25 <= 1.3]
    
    def process(exp_str):
        try:
            T = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days / 365.25
            chain = ticker.option_chain(exp_str)
            local = []
            # Standard OTM filtering for Heston calibration
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