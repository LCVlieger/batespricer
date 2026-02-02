import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import warnings
import os

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
    return NSSYieldCurve(curve_fit)

# ---------------------------------------------------------
#  VECTORIZED ALBRECHER (2007) STABLE PRICER
# ---------------------------------------------------------
class HestonAnalyticalPricer:
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, kappa, theta, xi, rho, v0):
        # 1. Integration Settings
        N_grid = 256
        u_max = 100.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] # (N_grid, 1)

        # q is usually 0 for SPX Index options unless specified
        q = 0.0 

        def heston_char_func(phi):
            # rho * xi * u * 1j - kappa
            # Albrecher (2007) Form
            d = np.sqrt((rho * xi * phi * 1j - kappa)**2 + xi**2 * (phi * 1j + phi**2))
            
            # g auxiliary
            g = (kappa - rho * xi * phi * 1j - d) / (kappa - rho * xi * phi * 1j + d)
            
            # Complex Exponents (C * v0 + D)
            # C exponent for v0
            C = (1/xi**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))) * (kappa - rho * xi * phi * 1j - d)
            
            # D exponent (Stable split logarithm)
            val_num = 1 - g * np.exp(-d * T)
            val_denom = 1 - g
            D = (kappa * theta / xi**2) * ((kappa - rho * xi * phi * 1j - d) * T - 2 * (np.log(val_num) - np.log(val_denom)))
            
            # Drift
            drift = 1j * phi * np.log(S0 * np.exp((r - q) * T))
            
            return np.exp(C * v0 + D + drift)

        # Integrands (Replicating your P1/P2 logic)
        # Integrand P1: Re[ exp(-i*u*ln(K)) * phi(u-i) / (i*u*S*e^(r-q)T) ]
        phi_p1 = heston_char_func(u - 1j)
        num_p1 = np.exp(-1j * u * np.log(K)) * phi_p1
        den_p1 = 1j * u * S0 * np.exp((r - q) * T)
        int_p1 = np.real(num_p1 / den_p1)

        # Integrand P2: Re[ exp(-i*u*ln(K)) * phi(u) / (i*u) ]
        phi_p2 = heston_char_func(u)
        num_p2 = np.exp(-1j * u * np.log(K)) * phi_p2
        den_p2 = 1j * u
        int_p2 = np.real(num_p2 / den_p2)

        # Integration
        P1 = 0.5 + (1/np.pi) * np.sum(int_p1 * du, axis=0)
        P2 = 0.5 + (1/np.pi) * np.sum(int_p2 * du, axis=0)

        # Price assembly
        price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        return np.maximum(price, 0.0)

# --- HESTON CALIBRATOR ---
class HestonCalibrator:
    def __init__(self, S0: float, r_curve: NSSYieldCurve):
        self.S0, self.r_curve = S0, r_curve

    def calibrate(self, options: List[MarketOption]) -> Dict:
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        
        # 5-Parameter Bounds (v0, kappa, theta, xi, rho)
        bounds = [(1e-3, 0.1), (1e-3, 5.0), (1e-3, 0.1), (1e-2, 1.0), (-1.0, 0.0)]
        x0 = [0.04, 2.5, 0.04, 0.5, -0.7]

        def objective(p):
            v0, k, th, xi, rho = p
            try:
                model_p = HestonAnalyticalPricer.price_european_call_vectorized(
                    self.S0, strikes, maturities, r_vec, k, th, xi, rho, v0
                )
                return np.mean((model_p - market_prices)**2)
            except: 
                return 1e9

        def callback(xk):
             print(f"==> [Step] RMSE: {np.sqrt(objective(xk)):.4f} | v0={xk[0]:.4f} k={xk[1]:.2f} th={xk[2]:.4f} xi={xk[3]:.4f} rho={xk[4]:.2f}")

        print(f"\nCalibrating Heston (5-Param Albrecher) on {len(options)} options...")
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, callback=callback, 
                       options={'eps': 1e-2, 'ftol': 1e-8})
        
        return {"v0": res.x[0], "kappa": res.x[1], "theta": res.x[2], 
                "xi": res.x[3], "rho": res.x[4], "rmse": np.sqrt(res.fun)}

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
    FRED_API_KEY = os.getenv("FRED_API_KEY") 
    TARGET_DATE = "2022-03-25"
    r_curve = fetch_treasury_rates_fred(TARGET_DATE, FRED_API_KEY)
    options, S0 = load_spx_replication("src/spx_eod_202203.txt", TARGET_DATE)
    print(f"Ready: S0={S0:.2f} | Options={len(options)}")
    cal = HestonCalibrator(S0, r_curve)
    res = cal.calibrate(options)
    print("\nFINAL RESULTS:")
    for k, v in res.items(): print(f"  {k}: {v:.6f}")