import os
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

BOX_SPREAD = 0.0045

@dataclass
class MarketOption:
    """Single market-observed option with mid price and bid-ask spread."""
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0

def save_options_to_cache(options: List[MarketOption], ticker: str):
    os.makedirs("cache", exist_ok=True)
    path = f"cache/options_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(path, "w") as f:
        json.dump([asdict(o) for o in options], f, indent=4)
    return path

def load_options_from_cache(filepath: str) -> List[MarketOption]:
    with open(filepath, "r") as f:
        return [MarketOption(**item) for item in json.load(f)]

class NSSYieldCurve:
    """Nelson-Siegel-Svensson interpolated yield curve from FRED."""
    def __init__(self, curve_fit, spread=0.0):
        self.curve = curve_fit
        self.spread = spread

    def get_rate(self, T: float) -> float:
        return float(self.curve(max(T, 1e-4))) + self.spread

    def to_dict(self):
        return {f"{round(t,3)}Y": self.get_rate(t) for t in [0.08, 0.25, 0.5, 1.0]}

def fetch_treasury_rates_fred(date_str: str, api_key: str) -> NSSYieldCurve:
    """Fetch UST rates from FRED and fit NSS curve via OLS."""
    series = {1/12: "DGS1MO", 3/12: "DGS3MO", 6/12: "DGS6MO", 1.0: "DGS1", 2.0: "DGS2"}
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    for i in range(6):
        d = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        mats, yields = [], []
        for tenor, s_id in series.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json&observation_start={d}&observation_end={d}"
                val = requests.get(url, timeout=3).json()['observations'][0]['value']
                if val != '.':
                    mats.append(tenor)
                    yields.append(float(val) / 100.0)
            except: continue
            
        if len(mats) >= 3:
            from nelson_siegel_svensson.calibrate import calibrate_nss_ols
            curve_fit, _ = calibrate_nss_ols(np.array(mats), np.array(yields))
            return NSSYieldCurve(curve_fit, spread=BOX_SPREAD)
    raise ValueError("FRED rates unavailable.")

def calculate_spx_time_to_maturity(expiry: datetime, ticker: str) -> float:
    now = datetime.now()
    am_settled = ["^SPX", "^NDX", "^VIX", "^RUT", "^GDAXI"]
    is_am = any(ticker.startswith(t) for t in am_settled)
    is_monthly = expiry.weekday() == 4 and 15 <= expiry.day <= 21

    set_time = dt_time(9, 30) if (is_am and is_monthly) else dt_time(16, 0)
    delta = datetime.combine(expiry.date(), set_time) - now
    return max(delta.total_seconds() / (365.25 * 24 * 3600), 1e-6)

class ImpliedDividendCurve:
    """Term structure of dividend yields implied from put-call parity."""
    def __init__(self, df: pd.DataFrame, S0: float, r_curve, ticker: str = ""):
        self.yields = {}
        is_index = ticker.startswith("^") or ticker in ["SPX", "NDX", "RUT"]
        
        fundamental_q = 0.0
        if not is_index:
            try:
                raw_q = yf.Ticker(ticker).info.get('dividendYield', 0) / 100
                fundamental_q = raw_q if 0 <= raw_q <= 0.15 else 0.0
            except: pass

        for T in sorted(df['T'].unique()):
            if is_index:
                sub = df[df['T'] == T]
                r = r_curve.get_rate(T)
                parity = (sub['C_MID'] - sub['P_MID']) + sub['STRIKE'] * np.exp(-r * T)
                intercept = np.median(parity)
                q = -np.log(intercept / S0) / T if (T > 1e-4 and intercept > 0) else 0.0
                self.yields[T] = float(np.clip(q, -0.01, 0.06))
            else:
                self.yields[T] = fundamental_q

        mats = np.sort(list(self.yields.keys()))
        vals = np.array([self.yields[m] for m in mats])
        self.interp = interp1d(mats, vals, kind='linear', bounds_error=False, fill_value=(vals[0], vals[-1]))
        self.min_T, self.max_T = mats[0], mats[-1]

    def get_rate(self, T: float) -> float:
        return float(self.interp(np.clip(T, self.min_T, self.max_T)))

def fetch_raw_data(ticker_symbol: str) -> pd.DataFrame:
    t_obj = yf.Ticker(ticker_symbol)
    exps = getattr(t_obj, 'options', [])
    if not exps: return pd.DataFrame()

    targets = [0.019, 0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.25]
    selected = {min(exps, key=lambda x: abs(calculate_spx_time_to_maturity(datetime.strptime(x, "%Y-%m-%d"), ticker_symbol) - t)) for t in targets}

    def fetch_one(exp_str):
        try:
            T = calculate_spx_time_to_maturity(datetime.strptime(exp_str, "%Y-%m-%d"), ticker_symbol)
            chain = t_obj.option_chain(exp_str)
            df_c = chain.calls[['strike', 'bid', 'ask']].rename(columns={'strike':'STRIKE', 'bid':'bC', 'ask':'aC'})
            df_p = chain.puts[['strike', 'bid', 'ask']].rename(columns={'strike':'STRIKE', 'bid':'bP', 'ask':'aP'})
            full = df_c.merge(df_p, on='STRIKE')
            full['C_MID'], full['P_MID'], full['T'] = (full['bC']+full['aC'])/2, (full['bP']+full['aP'])/2, T
            return full
        except: return None

    workers = 4 if ticker_symbol.startswith("^") else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [r for r in ex.map(fetch_one, sorted(list(selected))) if r is not None]
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()

def get_market_implied_spot(ticker_symbol: str, raw_df: pd.DataFrame, r_curve) -> float:
    t_obj = yf.Ticker(ticker_symbol)
    if not ticker_symbol.startswith("^"):
        try:
            p = t_obj.fast_info.get('last_price', 0)
            if p > 0: return float(p)
        except: pass

    if raw_df is not None and not raw_df.empty:
        a_T = min(raw_df['T'].unique(), key=lambda x: abs(x - 0.0833))
        sub = raw_df[raw_df['T'] == a_T]
        if len(sub) > 5:
            reg = LinearRegression().fit(sub['STRIKE'].values.reshape(-1, 1), (sub['C_MID'] - sub['P_MID']).values)
            if abs(reg.coef_[0]) > 1e-5:
                f_price = -reg.intercept_ / reg.coef_[0]
                q_base = 0.013 if ticker_symbol.startswith("^") else 0.0
                return float(f_price / np.exp((r_curve.get_rate(a_T) - q_base) * a_T))

    return float(t_obj.history(period="1d")['Close'].iloc[-1])

def fetch_options(ticker_symbol: str, S0: float, target_size: int = 300) -> List[MarketOption]:
    """Download OTM option chain and filter by liquidity."""
    if np.isnan(S0): return []
    t_obj = yf.Ticker(ticker_symbol)
    exps = [e for e in getattr(t_obj, 'options', []) if 0.04 <= calculate_spx_time_to_maturity(datetime.strptime(e, "%Y-%m-%d"), ticker_symbol) <= 1.3]

    def process(exp_str):
        try:
            T = calculate_spx_time_to_maturity(datetime.strptime(exp_str, "%Y-%m-%d"), ticker_symbol)
            chain, local = t_obj.option_chain(exp_str), []
            for typ, data, f in [('PUT', chain.puts, lambda k: k < S0*0.98), ('CALL', chain.calls, lambda k: k > S0*1.02)]:
                sub = data[f(data['strike']) & (data['strike'] > S0*0.7) & (data['strike'] < S0*1.3)]
                for _, r in sub.iterrows():
                    mid, bid, ask = (r['bid']+r['ask'])/2, r['bid'], r['ask']
                    if mid > 0.05 and bid > 0 and (ask-bid)/max(mid, 0.01) < 0.25:
                        local.append(MarketOption(r['strike'], T, mid, typ, bid, ask))
            return local
        except: return []

    workers = 4 if ticker_symbol.startswith("^") else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [item for sub in ex.map(process, exps) for item in sub]
    
    if len(res) > target_size:
        return [res[i] for i in np.linspace(0, len(res)-1, target_size, dtype=int)]
    return res