import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple
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

# ---------------------------------------------------------
#  SMART DIVIDEND CURVE (EXTRACTED FROM PUT-CALL PARITY)
# ---------------------------------------------------------
class ImpliedDividendCurve:
    """
    Extracts the implied dividend yield term structure from market prices
    using Put-Call Parity on ATM options: 
    C - P = S0 * exp(-qT) - K * exp(-rT)
    """
    def __init__(self, df: pd.DataFrame, S0: float, r_curve: NSSYieldCurve):
        self.yields = {}
        
        # 1. Group by Maturity
        for T in sorted(df['T'].unique()):
            subset = df[df['T'] == T]
            
            # 2. Find Strike closest to ATM Forward (proxy S0*exp(rT))
            r = r_curve.get_rate(T)
            F_approx = S0 * np.exp(r * T)
            
            # Filter for rows where both Call and Put mid prices exist
            valid_rows = subset.dropna(subset=['C_MID', 'P_MID'])
            if valid_rows.empty: continue
            
            best_idx = (valid_rows['STRIKE'] - F_approx).abs().idxmin()
            row = valid_rows.loc[best_idx]
            
            K = row['STRIKE']
            C = row['C_MID']
            P = row['P_MID']
            
            # 3. Solve Put-Call Parity for q: S0 * exp(-qT) = C - P + K * exp(-rT)
            rhs = C - P + K * np.exp(-r * T)
            
            if rhs > 0:
                q = -np.log(rhs / S0) / T
                self.yields[T] = q
            else:
                self.yields[T] = 0.015 # Robust fallback

    def get_rate(self, T: float) -> float:
        mats = sorted(self.yields.keys())
        if not mats: return 0.0
        # Use linear interpolation between extracted maturity points
        return np.interp(T, mats, [self.yields[m] for m in mats])

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
#  HIGH-SPEED VECTORIZED ALBRECHER (2007)
# ---------------------------------------------------------
class HestonAnalyticalPricer:
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        N_grid = 400  
        u_max = 100.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 

        T_mat = T[np.newaxis, :]  
        r_mat = r[np.newaxis, :]
        q_mat = q[np.newaxis, :] 
        K_mat = K[np.newaxis, :]

        def get_cf(phi):
            d = np.sqrt((rho * xi * phi * 1j - kappa)**2 + xi**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi * phi * 1j - d) / (kappa - rho * xi * phi * 1j + d)
            exp_neg_dT = np.exp(-d * T_mat)
            C = (1/xi**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT)) * (kappa - rho * xi * phi * 1j - d)
            D = (kappa * theta / xi**2) * ((kappa - rho * xi * phi * 1j - d) * T_mat - 
                2 * (np.log(1 - g * exp_neg_dT) - np.log(1 - g)))
            
            # Forward Price S0 * exp((r - q) * T)
            drift = 1j * phi * np.log(S0 * np.exp((r_mat - q_mat) * T_mat))
            return np.exp(C * v0 + D + drift)

        cf_p1 = get_cf(u - 1j)
        int_p1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p1) / (1j * u * S0 * np.exp((r_mat - q_mat) * T_mat)))
        cf_p2 = get_cf(u)
        int_p2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p2) / (1j * u))

        P1 = 0.5 + (1/np.pi) * np.sum(int_p1 * du, axis=0)
        P2 = 0.5 + (1/np.pi) * np.sum(int_p2 * du, axis=0)
        
        # BSM Discounting: S0*exp(-qT)*P1 - K*exp(-rT)*P2
        price = S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2
        return np.maximum(price.flatten(), 0.0)

# --- HESTON CALIBRATOR ---
class HestonCalibrator:
    def __init__(self, S0: float, r_curve: NSSYieldCurve, q_curve: ImpliedDividendCurve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def calibrate(self, options: List[MarketOption]) -> Dict:
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        bounds = [(1e-3, 0.1), (1e-3, 5.0), (1e-3, 0.1), (1e-2, 1.5), (-1.0, 0.0)]
        x0 = [0.04, 2.5, 0.04, 0.5, -0.7]

        def objective(p):
            v0, k, th, xi, rho = p
            try:
                model_p = HestonAnalyticalPricer.price_european_call_vectorized(
                    self.S0, strikes, maturities, r_vec, q_vec, k, th, xi, rho, v0
                )
                return np.mean((model_p - market_prices)**2)
            except: return 1e9

        def callback(xk):
             mse = objective(xk)
             print(f"==> [Step] RMSE: {np.sqrt(mse):.4f} | v0={xk[0]:.4f} k={xk[1]:.2f} th={xk[2]:.4f} xi={xk[3]:.4f} rho={xk[4]:.2f}")

        print(f"\nCalibrating Heston (Albrecher Vectorized) on {len(options)} options...")
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-9,
                       options={'eps': 1e-3, 'maxiter': 500})
        
        final_prices = HestonAnalyticalPricer.price_european_call_vectorized(
            self.S0, strikes, maturities, r_vec, q_vec, res.x[1], res.x[2], res.x[3], res.x[4], res.x[0]
        )
        mae = np.mean(np.abs(final_prices - market_prices))

        return {"v0": res.x[0], "kappa": res.x[1], "theta": res.x[2], 
                "xi": res.x[3], "rho": res.x[4], "rmse": np.sqrt(res.fun), "mae": mae}

# --- LOADER ---
def load_spx_replication(file_path, target_date) -> Tuple[List[MarketOption], pd.DataFrame, float]:
    df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.str.strip(' []')
    
    cols = ['STRIKE','C_BID','C_ASK','P_BID','P_ASK','UNDERLYING_LAST']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['STRIKE','UNDERLYING_LAST'])
    df['QUOTE_DATE'], df['EXPIRE_DATE'] = pd.to_datetime(df['QUOTE_DATE']), pd.to_datetime(df['EXPIRE_DATE'])
    
    day_data = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)].copy()
    S0 = day_data['UNDERLYING_LAST'].iloc[0]
    
    day_data['C_MID'] = (day_data['C_BID'] + day_data['C_ASK']) / 2
    if 'P_BID' in day_data.columns:
        day_data['P_MID'] = (day_data['P_BID'] + day_data['P_ASK']) / 2
    
    day_data['T'] = (day_data['EXPIRE_DATE'] - day_data['QUOTE_DATE']).dt.days / 365.25
    day_data = day_data[(day_data['T'] > 0.04) & (day_data['T'] < 2.5)]
    
    raw_df_for_curve = day_data.copy()

    # Filter for Calibration (Strikes 3200 to 4800)
    maturity_groups = day_data.groupby('T')['STRIKE'].apply(set)
    common_strikes = set.intersection(*maturity_groups.tolist())
    common_strikes = {k for k in common_strikes if 3200 < k < 4800} 
    calib_data = day_data[day_data['STRIKE'].isin(common_strikes)]
    
    calib_options = [MarketOption(row['STRIKE'], row['T'], row['C_MID']) for _, row in calib_data.iterrows()]
    
    return calib_options, raw_df_for_curve, S0

# --- CURVE PRINTER ---
def print_full_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Risk-Free (r)':<15} | {'Div Yield (q)':<15}")
    print("-" * 60)
    # Print standard financial tenors
    tenors = [1/12, 3/12, 6/12, 1.0, 2.0]
    labels = ["1 Month", "3 Month", "6 Month", "1 Year", "2 Year"]
    
    for t, label in zip(tenors, labels):
        r = r_curve.get_rate(t)
        q = q_curve.get_rate(t)
        print(f"{label:<10} | {r*100:>13.4f}% | {q*100:>13.4f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    FRED_API_KEY = os.getenv("FRED_API_KEY") 
    TARGET_DATE = "2022-03-25"
    
    options, raw_df, S0 = load_spx_replication("src/spx_eod_202203.txt", TARGET_DATE)
    r_curve = fetch_treasury_rates_fred(TARGET_DATE, FRED_API_KEY)
    q_curve = ImpliedDividendCurve(raw_df, S0, r_curve)
    
    # Printing the full curves
    print_full_curves(r_curve, q_curve)
    
    print(f"Ready: S0={S0:.2f} | Calibration Options={len(options)}")
    
    cal = HestonCalibrator(S0, r_curve, q_curve)
    res = cal.calibrate(options)
    
    print("\nFINAL RESULTS:")
    for k, v in res.items(): print(f"  {k}: {v:.6f}")