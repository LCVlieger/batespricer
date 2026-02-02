import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import warnings
import os

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY environment variable not set.")
# --- DATA STRUCTURES ---
@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL"

class NSSYieldCurve:
    def __init__(self, curve_fit):
        self.curve = curve_fit
    def get_rate(self, T: float) -> float:
        return float(self.curve(T))

# --- FRED RATE FETCHER ---
def fetch_treasury_rates_fred(date_str: str, api_key: str) -> NSSYieldCurve:
    print(f"Fetching Official Treasury Yields from FRED for {date_str}...")
    series_map = {
        1/12: "DGS1MO", 3/12: "DGS3MO", 6/12: "DGS6MO", 
        1.0: "DGS1", 2.0: "DGS2", 5.0: "DGS5", 
        10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30"
    }
    maturities, yields = [], []
    for tenor, series_id in series_map.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={date_str}&observation_end={date_str}"
        try:
            response = requests.get(url, timeout=10).json()
            val = response['observations'][0]['value']
            if val != '.':
                maturities.append(tenor)
                yields.append(float(val) / 100.0)
        except Exception: continue

    curve_fit, _ = calibrate_nss_ols(np.array(maturities), np.array(yields))
    nss_curve = NSSYieldCurve(curve_fit)
    return nss_curve

# --- HESTON CALIBRATOR ---
class HestonCalibrator:
    def __init__(self, S0: float, r_curve: NSSYieldCurve):
        self.S0, self.r_curve = S0, r_curve

    def calibrate(self, options: List[MarketOption]) -> Dict:
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        
        # --- EXACT NOTEBOOK BOUNDS ---
        bounds = [(1e-3, 0.1), (1e-3, 5.0), (1e-3, 0.1), (1e-2, 1.0), (-1.0, 0.0), (-1.0, 1.0)]
        
        # --- EXACT NOTEBOOK STARTING POINT ---
        x0 = [0.1, 3.0, 0.05, 0.3, -0.8, 0.03] 

        def objective(p):
            v0, k, th, sig, rho, lam = p
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model_p = _heston_price_nb(self.S0, strikes, maturities, r_vec, v0, k, th, sig, rho, lam)
                    return np.mean((model_p - market_prices)**2)
            except: return 1e9

        def callback(xk):
             print(f"   [Step] v0={xk[0]:.4f}, k={xk[1]:.2f}, th={xk[2]:.4f}, sig={xk[3]:.4f}, rho={xk[4]:.2f}, lam={xk[5]:.4f}")

        print(f"\nCalibrating Heston on {len(options)} options...")
        # eps and ftol added to ensure the solver actually moves from x0
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       callback=callback, tol=1e-4, options={'eps': 1e-3})
        
        return {"v0": res.x[0], "kappa": res.x[1], "theta": res.x[2], 
                "sigma": res.x[3], "rho": res.x[4], "lambd": res.x[5], "rmse": np.sqrt(res.fun)}

# --- PRICING HELPERS ---
def _heston_price_nb(S0, K, tau, r, v0, kappa, theta, sigma, rho, lambd):
    N, umax = 5000, 100 
    dphi = umax / N
    phi = (dphi * (2 * np.arange(1, N) + 1) / 2)[:, np.newaxis] 
    args = (S0, tau, r, v0, kappa, theta, sigma, rho, lambd)
    term1 = 0.5 * (S0 - K * np.exp(-r * tau))
    num = np.exp(r * tau) * _char_func_nb(phi - 1j, *args) - K * _char_func_nb(phi, *args)
    den = 1j * phi * (K**(1j * phi))
    integral = np.real(np.sum(dphi * num / den, axis=0))
    return term1 + integral / np.pi

def _char_func_nb(phi, S0, tau, r, v0, kappa, theta, sigma, rho, lambd):
    a, b = kappa * theta, kappa + lambd
    rspi = rho * sigma * phi * 1j
    d = np.sqrt((rspi - b)**2 + (phi * 1j + phi**2) * sigma**2)
    g = (b - rspi + d) / (b - rspi - d)
    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
    exp2 = np.exp(a * tau * (b - rspi + d) / sigma**2 + v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2)
    return exp1 * term2 * exp2

# --- LOADER ---
def load_spx_replication(file_path, target_date):
    df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.str.strip(' []')
    for c in ['STRIKE','C_BID','C_ASK','UNDERLYING_LAST']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['STRIKE','UNDERLYING_LAST'])
    df['QUOTE_DATE'], df['EXPIRE_DATE'] = pd.to_datetime(df['QUOTE_DATE']), pd.to_datetime(df['EXPIRE_DATE'])
    day_data = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)].copy()
    S0 = day_data['UNDERLYING_LAST'].iloc[0]
    day_data['T'] = (day_data['EXPIRE_DATE'] - day_data['QUOTE_DATE']).dt.days / 365.25
    day_data = day_data[(day_data['T'] > 0.04) & (day_data['T'] < 1.0)]
    maturity_groups = day_data.groupby('T')['STRIKE'].apply(set)
    common_strikes = set.intersection(*maturity_groups.tolist())
    common_strikes = {k for k in common_strikes if 3200 < k < 4800} 
    day_data = day_data[day_data['STRIKE'].isin(common_strikes)]
    return [MarketOption(row['STRIKE'], row['T'], (row['C_BID'] + row['C_ASK']) / 2) for _, row in day_data.iterrows()], S0

if __name__ == "__main__":
    TARGET_DATE = "2022-03-25"
    r_curve = fetch_treasury_rates_fred(TARGET_DATE, FRED_API_KEY)
    options, S0 = load_spx_replication("src/spx_eod_202203.txt", TARGET_DATE)
    print(f"Ready: S0={S0:.2f} | Options={len(options)}")
    cal = HestonCalibrator(S0, r_curve)
    res = cal.calibrate(options)
    print("\nFINAL RESULTS:")
    for k, v in res.items(): print(f"  {k}: {v:.6f}")