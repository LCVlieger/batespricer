import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict
from scipy.interpolate import interp1d
import warnings

# --- DATA STRUCTURES ---
@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 

class SimpleYieldCurve:
    def __init__(self, tenors: List[float], rates: List[float]):
        self.tenors = tenors
        self.rates = rates
        self.curve = interp1d(tenors, rates, kind='linear', fill_value="extrapolate")

    def get_rate(self, T: float) -> float:
        return float(self.curve(T)) if T > 1e-5 else float(self.rates[0])

# --- ANALYTICAL CALIBRATOR (Exact Notebook Replication) ---
class HestonCalibrator:
    def __init__(self, S0: float, r_curve: SimpleYieldCurve):
        self.S0 = S0
        self.r_curve = r_curve

    def calibrate(self, options: List[MarketOption]) -> Dict:
        strikes = np.array([opt.strike for opt in options])
        maturities = np.array([opt.maturity for opt in options])
        market_prices = np.array([opt.market_price for opt in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        
        # Exact Notebook Bounds: v0, kappa, theta, sigma, rho, lambd
        bounds = [(1e-3, 0.1), (1e-3, 5.0), (1e-3, 0.1), (1e-2, 1.0), (-1.0, 0.0), (-1.0, 1.0)]
        x0 = [0.1, 3.0, 0.05, 0.3, -0.8, 0.03] 

        def objective(params):
            v0, kappa, theta, sigma, rho, lambd = params
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model_prices = _heston_price_notebook_style(
                        self.S0, strikes, maturities, r_vec, 0.0,
                        v0, kappa, theta, sigma, rho, lambd
                    )
                    # MSE objective matches SqErr function in notebook
                    return np.mean((model_prices - market_prices)**2)
            except:
                return 1e9

        def callback(xk):
             print(f"   [Analytical] v0={xk[0]:.4f}, k={xk[1]:.2f}, theta={xk[2]:.4f}, sigma={xk[3]:.4f}, rho={xk[4]:.2f}, lambd={xk[5]:.4f}")

        print(f"Starting Calibration on {len(options)} options...")
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, callback=callback, tol=1e-4)

        return {
            "v0": result.x[0], "kappa": result.x[1], "theta": result.x[2],
            "sigma": result.x[3], "rho": result.x[4], "lambd": result.x[5],
            "fun": result.fun, "rmse": np.sqrt(result.fun)
        }

# --- HELPERS: EXACT NOTEBOOK FORMULAS ---
def _heston_price_notebook_style(S0, K, tau, r, q, v0, kappa, theta, sigma, rho, lambd):
    N, umax = 10000, 100 # Notebook integration parameters
    dphi = umax / N
    phi = dphi * (2 * np.arange(1, N) + 1) / 2
    phi = phi[:, np.newaxis] 

    args = (S0, tau, r, q, v0, kappa, theta, sigma, rho, lambd)
    term1 = 0.5 * (S0 * np.exp(-q * tau) - K * np.exp(-r * tau))
    
    numerator = np.exp(r * tau) * _char_func_notebook(phi - 1j, *args) - K * _char_func_notebook(phi, *args)
    denominator = 1j * phi * (K**(1j * phi))
    
    integral_sum = np.real(np.sum(dphi * numerator / denominator, axis=0))
    return term1 + integral_sum / np.pi

def _char_func_notebook(phi, S0, tau, r, q, v0, kappa, theta, sigma, rho, lambd):
    # Mathematical implementation matching Cell 7 in the reference notebook
    a, b = kappa * theta, kappa + lambd
    rspi = rho * sigma * phi * 1j
    d = np.sqrt((rspi - b)**2 + (phi * 1j + phi**2) * sigma**2)
    g = (b - rspi + d) / (b - rspi - d)
    exp1 = np.exp((r - q) * phi * 1j * tau)
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
    exp2 = np.exp(a * tau * (b - rspi + d) / sigma**2 + 
                  v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2)
    return exp1 * term2 * exp2

# --- REPLICATED LOADER ---
def load_spx_replication(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.str.strip(' []')
    for c in ['STRIKE','C_BID','C_ASK','UNDERLYING_LAST']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['STRIKE','UNDERLYING_LAST'])
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])
    
    day_data = df[df['QUOTE_DATE'] == "2022-03-25"].copy()
    S0 = day_data['UNDERLYING_LAST'].iloc[0]
    day_data['T'] = (day_data['EXPIRE_DATE'] - day_data['QUOTE_DATE']).dt.days / 365.25
    
    # 1. Filter Maturity: (0.04 < T < 1.0)
    day_data = day_data[(day_data['T'] > 0.04) & (day_data['T'] < 1.0)]
    
    # 2. THE NOTEBOOK "CHEAT": Intersection of Strikes
    # Keep only strikes that appear on every maturity date to eliminate "noisy" wings.
    maturity_groups = day_data.groupby('T')['STRIKE'].apply(set)
    common_strikes = set.intersection(*maturity_groups.tolist())
    
    # 3. Filter for Strike Range (3000 to 5000)
    common_strikes = {k for k in common_strikes if 3000 < k < 5000}
    day_data = day_data[day_data['STRIKE'].isin(common_strikes)]
    
    options = []
    # 4. Use CALLS only
    for _, row in day_data.iterrows():
        mid = (row['C_BID'] + row['C_ASK']) / 2
        options.append(MarketOption(row['STRIKE'], row['T'], mid, "CALL"))
        
    print(f"Loaded {len(options)} options (Intersection Mode)")
    return options, S0

# --- EXECUTION ---
def main():
    # Exact yields used in the notebook (Cell 16)
    tenors = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = np.array([0.15, 0.27, 0.50, 0.93, 1.52, 2.13, 2.32, 2.34, 2.37, 2.32, 2.65, 2.52]) / 100
    r_curve = SimpleYieldCurve(tenors, rates)
    
    options, S0 = load_spx_replication("src/spx_eod_202203.txt")
    print(f"Backtest Ready: S0={S0:.2f} | Options={len(options)}")
    
    cal = HestonCalibrator(S0, r_curve)
    res = cal.calibrate(options)
    
    print("\nFINAL CALIBRATED PARAMETERS:")
    for k, v in res.items(): print(f"  {k}: {v:.6f}")

if __name__ == "__main__":
    main()