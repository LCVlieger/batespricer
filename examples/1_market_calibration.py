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
from scipy.stats import norm

# --- IMPORTS ---
try:
    # Ensure you have updated calibration.py to include BatesCalibrator
    from heston_pricer.calibration import BatesCalibrator 
    # Ensure you have updated analytics.py (or bates_analytics.py) to include BatesAnalyticalPricer
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
    from heston_pricer.data import ImpliedDividendCurve, fetch_treasury_rates_fred, get_market_implied_spot, fetch_raw_data, fetch_options
except ImportError:
    print("Warning: Ensure 'heston_pricer' is in your PYTHONPATH.")

# =================================================================
# 3. SAVING & VALIDATION 
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_ana, options):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_{ticker}_{timestamp}"
    
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, label in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, label in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments...")
    
    # --- 1. PRE-COMPUTE WEIGHTS (Matching Calibrator logic) ---
    # We use the same inverse spread logic here to calculate Weighted RMSE
    raw_weights = []
    for opt in options:
        spread = max(abs(opt.ask - opt.bid), 0.01)
        raw_weights.append(1.0 / spread)
    
    normalized_weights = np.array(raw_weights) / np.mean(raw_weights)

    # --- 2. CALCULATE EXACT ANALYTICAL PRICES & ERRORS ---
    rows = []
    
    # Helper to unpack 8 Bates parameters safely
    def get_p(res): 
        keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
        return [res[k] for k in keys]

    p_ana_params = get_p(res_ana)

    weighted_sq_err_ana = []
    
    # Vectorize the validation pass for speed (instead of loop)
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    mkt_prices = np.array([o.market_price for o in options])
    r_vec = np.array([r_curve.get_rate(t) for t in mats])
    q_vec = np.array([q_curve.get_rate(t) for t in mats])

    # Price all at once using the vectorized engine
    model_prices = BatesAnalyticalPricer.price_european_call_vectorized(
        S0, strikes, mats, r_vec, q_vec, *p_ana_params
    )

    for i, opt in enumerate(options):
        r, q = r_vec[i], q_vec[i]
        model_p = model_prices[i]
        
        err_ana = model_p - opt.market_price
        weighted_sq_err_ana.append((err_ana * normalized_weights[i])**2)
        
        # Calculate Implied Vol only for reporting
        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, "CALL")
        
        rows.append({
            "T": round(opt.maturity, 3), 
            "K": opt.strike, 
            "Market": round(opt.market_price, 2),
            "Model": round(model_p, 2), 
            "Err": round(err_ana, 2),
            "IV_Mkt": round(iv_mkt, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    # --- 3. DERIVE FINAL METRICS ---
    rmse_ana = np.sqrt((df["Err"]**2).mean())
    w_obj_ana = np.sqrt(np.mean(weighted_sq_err_ana))

    meta = {
        "model": "Bates",
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample},
        "analytical": res_ana,
    }
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    
    print("\n" + "="*80)
    print(f"VALIDATION SAMPLE (First 10) | S0: {S0:.2f} | Yield Used: {q_curve.get_rate(1.0):.4%}")
    print("-" * 80)
    print(df.head(10).to_string(index=False))
    print("-" * 80)
    
    mean_price = df["Market"].mean()
    print(f"Analytical RMSE: {rmse_ana:.4f} ({rmse_ana/mean_price:.2%} of avg price)")
    print(f"Weighted Objective: {w_obj_ana:.4f}")
    print("="*80)
    print(f"Saved results to: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Risk-Free (r)':<15} | {'Div Yield (q)':<15}")
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
    ticker = "^SPX"
    
    print(f"Initializing Bates Calibration for {ticker}...")
    
    # 1. Data Fetching
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    raw_df = fetch_raw_data(ticker)
    S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve)
    print_curves(r_curve, q_curve)
    
    options_raw = fetch_options(ticker, S0_actual, target_size=300)
    
    # 2. Processing (Synthetic Calls)
    options_processed = []
    print(f"Processing {len(options_raw)} options (Synthetic Calls)...")
    for opt in options_raw:
        r_T, q_T = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
        if opt.option_type == "PUT":
            # Put-Call Parity conversion
            price = opt.market_price + (S0_actual * np.exp(-q_T * opt.maturity) - opt.strike * np.exp(-r_T * opt.maturity))
            opt.market_price, opt.option_type = price, "CALL"
        options_processed.append(opt)
    
    # 3. Calibration
    print(f"\n{'='*20} BATES ANALYTICAL CALIBRATION {'='*20}")
    t0 = time.time()
    
    # Switch to BatesCalibrator
    calib_analytic = BatesCalibrator(S0=S0_actual, r_curve=r_curve, q_curve=q_curve)
    res_a = calib_analytic.calibrate(options_processed)
    
    print(f"CALIBRATION RESULTS (Time: {time.time()-t0:.2f}s)")
    print(f"Obj (Weighted): {res_a.get('weighted_obj', 0):.4f} | RMSE (Price): {res_a.get('rmse', 0):.4f}")
    
    # Print all 8 parameters clearly
    print("-" * 60)
    print(f"Heston Component:")
    print(f"  v0:    {res_a['v0']:.4f}")
    print(f"  kappa: {res_a['kappa']:.4f}")
    print(f"  theta: {res_a['theta']:.4f}")
    print(f"  xi:    {res_a['xi']:.4f}")
    print(f"  rho:   {res_a['rho']:.4f}")
    print(f"Jump Component (Merton):")
    print(f"  lambda:  {res_a['lamb']:.4f}  (Jumps per year)")
    print(f"  mu_J:    {res_a['mu_j']:.4f}  (Mean log-jump size)")
    print(f"  sigma_J: {res_a['sigma_j']:.4f}  (Jump size volatility)")
    print("-" * 60)

    # 4. Saving
    save_results(ticker, S0_actual, r_curve, q_curve, res_a, options_processed)

if __name__ == "__main__":
    main()