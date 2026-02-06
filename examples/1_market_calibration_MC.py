import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from heston_pricer.calibration import BatesCalibratorMC 
from heston_pricer.analytics import implied_volatility
from heston_pricer.data import (
    fetch_treasury_rates_fred, 
    fetch_raw_data, 
    fetch_options, 
    get_market_implied_spot, 
    ImpliedDividendCurve
)

# =================================================================
# 3. SAVING & VALIDATION (Synchronized with Analytical)
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_params, options, mc_calibrator):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_MC_{ticker}_{timestamp}"
    
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, label in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, label in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments using Final MC Parameters...")

    # --- 1. PRE-COMPUTE WEIGHTS (Matching Analytical Logic) ---
    # Note: MC usually uses spread weights, but we match the analytical file's 1.0 logic here for consistency
    raw_weights = [1.0 for _ in options]
    normalized_weights = np.array(raw_weights) / np.mean(raw_weights)

    # --- 2. GET MC PRICES ---
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    if isinstance(res_params, dict):
        p_values = [res_params[k] for k in keys]
    else:
        p_values = list(res_params.x)

    # Run high-precision pass for final saving (50k paths)
    mc_calibrator.n_paths = 50000 
    mc_calibrator._precompute(options) # Re-init noise/indices for new path count
    
    # get_prices returns (model, market, weights) - we ignore returned weights to match analytical file logic
    model_prices, _, _ = mc_calibrator.get_prices(p_values)

    rows = []
    weighted_sq_err = []
    
    # Vectorized lookups for IV calculation
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    r_vec = np.array([r_curve.get_rate(t) for t in mats])
    q_vec = np.array([q_curve.get_rate(t) for t in mats])

    for i, opt in enumerate(options):
        r, q = r_vec[i], q_vec[i]
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

    meta = {
        "model": "Bates_MC",
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample},
        "parameters": res_params,
        "notes": "Calibrated directly via Monte Carlo"
    }
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    
    print("\n" + "="*80)
    print(f"VALIDATION SAMPLE (First 10) | S0: {S0:.2f}")
    print("-" * 80)
    print(df.head(10).to_string(index=False))
    print("-" * 80)
    
    mean_price = df["Market"].mean()
    print(f"Final MC RMSE: {rmse:.4f} ({rmse/mean_price:.2%} of avg price)")
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
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    target_date = datetime.now().strftime("%Y-%m-%d")
    ticker = "NVDA" 
    
    print(f"Initializing Bates MC Calibration for {ticker}...")
    
    # 1. Data Fetching
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    raw_df = fetch_raw_data(ticker)
    S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve, ticker)
    print_curves(r_curve, q_curve)
    
    # Load Options
    options_processed = fetch_options(ticker, S0_actual, target_size=150) # Reduced size for speed
    print(f"Processing {len(options_processed)} options for MC Calibration...")
    
    # 2. Setup MC Calibrator
    # Note: Fewer paths/steps for speed during optimization; validation bumps this up
    mc_calib = BatesCalibratorMC(S0=S0_actual, r_curve=r_curve, q_curve=q_curve, n_paths=5000, n_steps=4000)
    mc_calib._precompute(options_processed)
    
    # 3. Optimization Setup
    print(f"\n{'='*20} STARTING MC OPTIMIZATION {'='*20}")
    print("WARNING: This will take significantly longer than analytical calibration.")
    
    # Bounds: k, theta, xi, rho, v0, lamb, mu_j, sigma_j
    # Revised bounds for stability
    bounds = [
        (0.1, 10.0),   # kappa 
        (0.001, 0.5),  # theta 
        (0.01, 1.5),   # xi (Capped for stability)
        (-0.99, -0.0), # rho 
        (0.005, 0.5),  # v0 (Floor raised to avoid clipping)
        (0.0, 5.0),    # lamb 
        (-0.5, 0.0),   # mu_j (Forced negative for SPX crashes)
        (0.01, 0.5)    # sigma_j 
    ]
    
    # Initial Guess (Standard Bates)
    x0 = [2.0, 0.04, 0.6, -0.7, 0.04, 0.1, -0.1, 0.1]
    
    def objective(p):
        try:
            # Get prices from MC engine
            model_p, market_p, weights = mc_calib.get_prices(p)
            
            # Weighted RMSE
            diff = (model_p - market_p)
            w_rmse = np.sqrt(np.mean((diff * weights)**2))
            return w_rmse
            
        except Exception as e:
            return 1e10

    def callback(xk):
        obj_val = objective(xk)
        print(f" [MC-Iter] Obj: {obj_val:10.4f} | "
      f"v0:{xk[4]:.4f} th:{xk[1]:.4f} ka:{xk[0]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f} | "
      f"Jumps(L:{xk[5]:.3f} mu:{xk[6]:.3f} sig:{xk[7]:.3f})")
        
    t0 = time.time()
    
    # Using SLSQP as it respects bounds strictly
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        callback=callback, 
        tol=1e-8,
        options={'maxiter': 50, 'eps': 1e-2} 
    )
    
    print(f"CALIBRATION DONE (Time: {time.time()-t0:.2f}s)")
    
    final_params = dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x))
    
    # Detailed Parameter Output (Matching Analytical Style)
    print("-" * 60)
    print(f"Heston Component:")
    print(f"  v0:      {final_params['v0']:.4f}")
    print(f"  kappa:   {final_params['kappa']:.4f}")
    print(f"  theta:   {final_params['theta']:.4f}")
    print(f"  xi:      {final_params['xi']:.4f}")
    print(f"  rho:     {final_params['rho']:.4f}")
    print(f"Jump Component (Merton):")
    print(f"  lambda:   {final_params['lamb']:.4f}  (Jumps per year)")
    print(f"  mu_J:     {final_params['mu_j']:.4f}  (Mean log-jump size)")
    print(f"  sigma_J:  {final_params['sigma_j']:.4f}  (Jump size volatility)")
    print("-" * 60)

    # 4. Saving & Verification
    save_results(ticker, S0_actual, r_curve, q_curve, final_params, options_processed, mc_calib)

if __name__ == "__main__":
    main()