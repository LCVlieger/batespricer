import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from heston_pricer.calibration import BatesCalibratorMC 
from heston_pricer.analytics import implied_volatility

# =================================================================
# HELPER CLASSES (To mimic objects from data fetchers)
# =================================================================
class InterpolatedCurve:
    """Reconstructs a curve from the JSON samples."""
    def __init__(self, curve_dict):
        # Parse keys like "0.0200Y" -> 0.02
        times = sorted([float(k.replace('Y', '')) for k in curve_dict.keys()])
        rates = [curve_dict[f"{t:.4f}Y"] for t in times]
        # Linear interpolation with flat extrapolation
        self.interp = interp1d(times, rates, kind='linear', fill_value=(rates[0], rates[-1]), bounds_error=False)

    def get_rate(self, T):
        return float(self.interp(T))

class SimpleOption:
    """Structure to hold option data compatible with BatesCalibratorMC"""
    def __init__(self, T, K, type_str, price, spread):
        self.maturity = T
        self.strike = K
        self.option_type = type_str
        self.market_price = price
        # Reconstruct Ask/Bid so the MC engine calculates the same weights (1/spread)
        half_spread = spread / 2.0
        self.ask = price + half_spread
        self.bid = price - half_spread
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from heston_pricer.calibration import BatesCalibratorMC 
# Added price_bates for the analytical validation step
from heston_pricer.analytics import implied_volatility, BatesAnalyticalPricer

# =================================================================
# HELPER CLASSES (To mimic objects from data fetchers)
# =================================================================
class InterpolatedCurve:
    """Reconstructs a curve from the JSON samples."""
    def __init__(self, curve_dict):
        # Parse keys like "0.0200Y" -> 0.02
        times = sorted([float(k.replace('Y', '')) for k in curve_dict.keys()])
        rates = [curve_dict[f"{t:.4f}Y"] for t in times]
        # Linear interpolation with flat extrapolation
        self.interp = interp1d(times, rates, kind='linear', fill_value=(rates[0], rates[-1]), bounds_error=False)

    def get_rate(self, T):
        return float(self.interp(T))

class SimpleOption:
    """Structure to hold option data compatible with BatesCalibratorMC"""
    def __init__(self, T, K, type_str, price, spread):
        self.maturity = T
        self.strike = K
        self.option_type = type_str
        self.market_price = price
        # Reconstruct Ask/Bid so the MC engine calculates the same weights (1/spread)
        half_spread = spread / 2.0
        self.ask = price + half_spread
        self.bid = price - half_spread

# =================================================================
# 3. SAVING & VALIDATION (Analytic Pricing on MC Params)
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_params, options, mc_calibrator):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_MC_{ticker}_{timestamp}"
    
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, label in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, label in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments ANALYTICALLY using Final MC Parameters...")

    # --- 1. PRE-COMPUTE WEIGHTS (Matching Analytical Logic) ---
    raw_weights = [1.0 for _ in options]
    normalized_weights = np.array(raw_weights) / np.mean(raw_weights)

    # --- 2. GET ANALYTICAL PRICES (Vectorized Sanity Check) ---
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    if isinstance(res_params, dict):
        p_values = [res_params[k] for k in keys]
    else:
        p_values = list(res_params.x)
        
    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p_values

    # Prepare vectorized inputs
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    types = np.array([o.option_type for o in options])
    r_T = np.array([r_curve.get_rate(o.maturity) for o in options])
    q_T = np.array([q_curve.get_rate(o.maturity) for o in options])

    # Batch pricing call
    model_prices = BatesAnalyticalPricer.price_vectorized(
        S0, strikes, mats, r_T, q_T, types,
        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
    )

    rows = []
    weighted_sq_err = []
    
    for i, opt in enumerate(options):
        r, q = r_T[i], q_T[i]
        model_p = model_prices[i]
        
        # Error Calculation
        err = model_p - opt.market_price
        weighted_sq_err.append((err * normalized_weights[i])**2)
        
        # Calculate IV using Black-Scholes inverter
        try:
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, opt.option_type)
            iv_model = implied_volatility(model_p, S0, opt.strike, opt.maturity, r, q, opt.option_type)
        except:
            iv_mkt, iv_model = 0.0, 0.0
        
        rows.append({
            "T": round(opt.maturity, 3), 
            "K": opt.strike, 
            "Spread": opt.ask - opt.bid,
            "Weight": 1/max(abs(opt.ask - opt.bid), 0.01),
            "Type": opt.option_type,
            "Market": round(opt.market_price, 2),
            "Model": round(model_p, 2), 
            "Err": round(err, 2),
            "IV_Mkt": round(iv_mkt, 4),
            "IV_Model": round(iv_model, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    rmse = np.sqrt((df["Err"]**2).mean())
    w_obj = np.sqrt(np.mean(weighted_sq_err))
    
    # Save parameters dict
    res_dict = dict(zip(keys, p_values))
    res_dict['rmse'] = rmse

    meta = {
        "model": "Bates_MC_Validation_Analytic",
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample},
        "parameters": res_dict,
        "notes": "Calibrated via MC, Verified via Analytic Formula"
    }
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    
    print("\n" + "="*80)
    print(f"VALIDATION SAMPLE (First 10) | S0: {S0:.2f}")
    print("-" * 80)
    print(df.head(10).to_string(index=False))
    print("-" * 80)
    
    mean_price = df["Market"].mean()
    print(f"Final Analytic RMSE (on MC params): {rmse:.4f} ({rmse/mean_price:.2%} of avg price)")
    print(f"Weighted Objective: {w_obj:.4f}")
    print("="*80)
    print(f"Saved results to: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Market Rate (r)':<15} | {'Div Yield (q)':<15}")
    print("-" * 60)
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]

    for t, label in tenors:
        print(f"{label:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")
    print("="*60 + "\n")


# =================================================================
# 4. MAIN EXECUTION
# =================================================================
def main():
    # --- FILE CONFIGURATION ---
    # Filenames updated to match your uploaded files
    json_path = "calibration_Analytic_^SPX_20260208_022951_meta.json"
    csv_path  = "calibration_Analytic_^SPX_20260208_022951_prices.csv"
    ticker = "^SPX" 
    
    print(f"Initializing Bates MC Calibration for {ticker} using LOCAL FILES...")
    
    # --- 1. Data Loading ---
    with open(json_path, 'r') as f:
        meta_data = json.load(f)

    # Reconstruct Market Data
    S0_actual = meta_data['market']['S0']
    r_curve = InterpolatedCurve(meta_data['market']['r_sample'])
    q_curve = InterpolatedCurve(meta_data['market']['q_sample'])
    
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    print_curves(r_curve, q_curve)
    
    # Load Options from CSV
    df_prices = pd.read_csv(csv_path)
    options_processed = []
    
    for _, row in df_prices.iterrows():
        opt = SimpleOption(
            T=row['T'],
            K=row['K'],
            type_str=row['Type'],
            price=row['Market'],
            spread=row['Spread']
        )
        options_processed.append(opt)

    print(f"Processing {len(options_processed)} options for MC Calibration...")
    
    # 2. Setup MC Calibrator
    mc_calib = BatesCalibratorMC(S0=S0_actual, r_curve=r_curve, q_curve=q_curve, n_paths=5000, n_steps=4000)
    mc_calib._precompute(options_processed)
    
    # 3. Optimization Setup
    print(f"\n{'='*20} STARTING MC OPTIMIZATION {'='*20}")

    short_opts = [o for o in options_processed if o.maturity < 0.1]
    closest = min(short_opts, key=lambda x: abs(x.strike - S0_actual))

    implied_v0_root = implied_volatility(closest.market_price, S0_actual, closest.strike, closest.maturity, r_curve.get_rate(closest.maturity), q_curve.get_rate(closest.maturity), closest.option_type)
    fixed_v0 = implied_v0_root ** 2
    v0_min, v0_max = fixed_v0 * 0.95, fixed_v0 * 1.05
    
    print(f"LOCKING v0 to Market Reality: {fixed_v0:.4f} (Vol: {implied_v0_root:.1%})")

    bounds = [
        (0.5, 10.0),   # kappa
        (0.01, 0.10),  # theta
        (0.1, 1.5),    # xi
        (-0.99, -0.5), # rho
        (0.005, 0.10), # v0
        (0.1, 3.0),    # lambda
        (-0.3, -0.01), # mu_j
        (0.01, 0.3)    # sigma_j
    ]
    
    x0 = [2.0, fixed_v0, 1.0, -0.7, fixed_v0, 0.1, -0.1, 0.1]
    
    def objective(p):
        try:
            model_p, market_p, weights = mc_calib.get_prices(p)
            diff = (model_p - market_p)
            w_rmse = np.sqrt(np.mean((diff * weights)**2))
            return w_rmse
        except Exception:
            return 1e10

    def callback(xk):
        obj_val = objective(xk)
        print(f" [MC-Iter] Obj: {obj_val:10.4f} | "
              f"v0:{xk[4]:.4f} th:{xk[1]:.4f} ka:{xk[0]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f}")
        
    t0 = time.time()
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, callback=callback, tol=1e-8, options={'maxiter': 50, 'eps': 1e-2})
    
    print(f"CALIBRATION DONE (Time: {time.time()-t0:.2f}s)")
    
    final_params = dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x))
    
    save_results(ticker, S0_actual, r_curve, q_curve, final_params, options_processed, mc_calib)
    
if __name__ == "__main__":
    main()