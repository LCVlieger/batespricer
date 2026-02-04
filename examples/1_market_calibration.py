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
from sklearn.linear_model import LinearRegression
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# --- IMPORTS FROM YOUR PACKAGE ---
try:
    from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC
    from heston_pricer.analytics import HestonAnalyticalPricer, implied_volatility
    from heston_pricer.market import MarketEnvironment
    from heston_pricer.data import ImpliedDividendCurve, fetch_treasury_rates_fred, get_market_implied_spot, fetch_raw_data, fetch_options
    from nelson_siegel_svensson.calibrate import calibrate_nss_ols
except ImportError:
    print("Warning: Ensure 'heston_pricer' is in your PYTHONPATH.")



# =================================================================
# 3. SAVING & VALIDATION (RE-PRICING MC FOR REPORT TRUTH)
# =================================================================
def calculate_bs_vega(S, K, T, r, q, sigma=0.20):
    if T <= 1e-5 or sigma <= 0: return 0.01
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return max(S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1), S * 0.0001)

# =================================================================
# 3. SAVING & VALIDATION (RE-EVALUATING WEIGHTED OBJ)
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_ana, res_mc, options):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_{ticker}_{timestamp}"
    
    tenors = [(0.0385, "2 Weeks"), (0.0833, "1 Month"), 
              (0.1667, "2 Months"), (0.25, "3 Months"), (0.3333, "4 Months"), 
              (0.4167, "5 Months"), (0.5, "6 Months"), (0.5833, "7 Months"), 
              (0.6667, "8 Months"), (0.75, "9 Months"), (0.8333, "10 Months"), 
              (0.9167, "11 Months"), (1.0, "1 Year")]
    
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, label in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, label in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments...")
    
    # --- 1. PRE-COMPUTE WEIGHTS (Matching Calibrator logic) ---
    raw_weights = []
    for opt in options:
        r, q = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
        vega = 1 #calculate_bs_vega(S0, opt.strike, opt.maturity, r, q, 0.20)
        spread = max(abs(opt.ask - opt.bid), 0.01)
        raw_weights.append(1.0 / (spread * vega))
    
    normalized_weights = np.array(raw_weights) / np.mean(raw_weights)

    # --- 2. CALCULATE EXACT ANALYTICAL PRICES & ERRORS ---
    rows = []
    def get_p(res): return [res[k] for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    p_ana_params = get_p(res_ana)
    p_mc_params = get_p(res_mc)

    weighted_sq_err_mc = []
    weighted_sq_err_ana = []
    for i, opt in enumerate(options):
        r, q = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
        
        # Fair comparison: Price BOTH sets of parameters with the Analytical Engine
        model_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *p_ana_params)
        model_mc_exact = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *p_mc_params)
        iv = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, "CALL")
        
        err_ana = model_ana - opt.market_price
        err_mc = model_mc_exact - opt.market_price
        weighted_sq_err_ana.append((err_ana * normalized_weights[i])**2)
        weighted_sq_err_mc.append((err_mc * normalized_weights[i])**2)
        rows.append({
            "T": round(opt.maturity, 3), "K": opt.strike, "Market": round(opt.market_price, 2),
            "Ana_Price": round(model_ana, 2), "Ana_Err": round(model_ana - opt.market_price, 2),
            "MC_Price": round(model_mc_exact, 2), "MC_Err": round(err_mc, 2),
            "IV_Mkt": round(iv, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    # --- 3. DERIVE FINAL METRICS ---
    rmse_ana = np.sqrt((df["Ana_Err"]**2).mean())
    rmse_mc_true = np.sqrt((df["MC_Err"]**2).mean())
    w_obj_ana = np.sqrt(np.mean(weighted_sq_err_ana))
    w_obj_mc_exact = np.sqrt(np.mean(weighted_sq_err_mc))
    # Store "True" analytical weighted objective in the meta
    res_mc_fixed = res_mc.copy()
    res_mc_fixed['weighted_obj_internal'] = res_mc.get('weighted_obj') # The noisy simulation score
    res_mc_fixed['weighted_obj'] = float(w_obj_mc_exact)              # The clean analytical score
    res_mc_fixed['rmse'] = float(rmse_mc_true)

    meta = {
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample},
        "analytical": res_ana,
        "monte_carlo": res_mc_fixed
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
    print(f"Analytical Weighted Obj: {w_obj_ana:.4f} ({w_obj_ana/mean_price:.2%} of avg price)")
    print(f"Monte Carlo RMSE (True): {rmse_mc_true:.4f} ({rmse_mc_true/mean_price:.2%} of avg price)")
    print(f"Monte Carlo Weighted Obj (True): {w_obj_mc_exact:.4f}")
    print("="*80)
    print(f"Saved results to: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Risk-Free (r)':<15} | {'Div Yield (q)':<15}")
    print("-" * 60)
    tenors = [(0.0385, "2 Weeks"), (0.0833, "1 Month"), 
              (0.1667, "2 Months"), (0.25, "3 Months"), (0.3333, "4 Months"), 
              (0.4167, "5 Months"), (0.5, "6 Months"), (0.5833, "7 Months"), 
              (0.6667, "8 Months"), (0.75, "9 Months"), (0.8333, "10 Months"), 
              (0.9167, "11 Months"), (1.0, "1 Year")]

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
    print(f"Initializing Calibration for {ticker}...")
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    raw_df = fetch_raw_data(ticker)
    S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve)
    print_curves(r_curve, q_curve)
    options_raw = fetch_options(ticker, S0_actual, target_size=300)
    options_processed = []
    print(f"Processing {len(options_raw)} options (Synthetic Calls)...")
    for opt in options_raw:
        r_T, q_T = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
        if opt.option_type == "PUT":
            price = opt.market_price + (S0_actual * np.exp(-q_T * opt.maturity) - opt.strike * np.exp(-r_T * opt.maturity))
            opt.market_price, opt.option_type = price, "CALL"
        options_processed.append(opt)
    print(f"\n{'='*20} 1. ANALYTICAL CALIBRATION (Albrecher) {'='*20}")
    t0 = time.time()
    calib_analytic = HestonCalibrator(S0=S0_actual, r_curve=r_curve, q_curve=q_curve)
    res_a = calib_analytic.calibrate(options_processed)
    print(f"ANALYTICAL RESULTS (Time: {time.time()-t0:.2f}s)")
    print(f"Obj (Weighted): {res_a.get('weighted_obj', 0):.4f} | RMSE (Price): {res_a.get('rmse', 0):.4f}")
    print(f"k: {res_a['kappa']:.4f} | th: {res_a['theta']:.4f} | xi: {res_a['xi']:.4f} | rho: {res_a['rho']:.4f} | v0: {res_a['v0']:.4f}\n")
    print(f"{'='*20} 2. MONTE CARLO CALIBRATION {'='*20}")
    t1 = time.time()
    calib_mc = HestonCalibratorMC(S0=S0_actual, r_curve=r_curve, q_curve=q_curve, n_paths=7500, n_steps=3000)
    x0 = [res_a[k] for k in ['kappa','theta','xi','rho','v0']]
    res_mc = calib_mc.calibrate(options_processed)
    print(f"MONTE CARLO RESULTS (Time: {time.time()-t1:.2f}s)")
    print(f"Obj (Weighted): {res_mc.get('weighted_obj', 0):.4f} | RMSE (Internal): {res_mc.get('rmse', 0):.4f}")
    print(f"k: {res_mc['kappa']:.4f} | th: {res_mc['theta']:.4f} | xi: {res_mc['xi']:.4f} | rho: {res_mc['rho']:.4f} | v0: {res_mc['v0']:.4f}\n")
    save_results(ticker, S0_actual, r_curve, q_curve, res_a, res_mc, options_processed)

if __name__ == "__main__":
    main()