import os
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from sklearn.linear_model import LinearRegression
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

# --- IMPORTS FROM YOUR PACKAGE ---
try:
    from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC
    from heston_pricer.analytics import HestonAnalyticalPricer
    from heston_pricer.market import MarketEnvironment
    from heston_pricer.data import ImpliedDividendCurve, fetch_treasury_rates_fred, get_market_implied_spot, fetch_raw_data, fetch_options
    from nelson_siegel_svensson.calibrate import calibrate_nss_ols
except ImportError:
    print("Warning: Ensure 'heston_pricer' is in your PYTHONPATH.")

# =================================================================
# 1. UTILITIES (IV & VALIDATION)
# =================================================================
def implied_volatility(price, S, K, T, r, q, option_type="CALL"):
    if price <= 0: return 0.0
    intrinsic = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0) if option_type == "PUT" else max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if price < intrinsic: return 0.0
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        val = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)) if option_type == "PUT" else (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return val - price
    try: return brentq(bs_price, 0.001, 5.0)
    except: return 0.0

# =================================================================
# 3. SAVING & VALIDATION
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_ana, res_mc, options):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_{ticker}_{timestamp}"
    
    meta = {
        "market": {"S0": S0, "r_sample": r_curve.to_dict(), "q_sample": q_curve.to_dict()},
        "analytical": res_ana,
        "monte_carlo": res_mc
    }
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n[Validation] Re-pricing {len(options)} instruments...")
    rows = []
    def get_p(res): return [res[k] for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    p_ana_params = get_p(res_ana)
    p_mc_params = get_p(res_mc)

    for opt in options:
        r = r_curve.get_rate(opt.maturity)
        q = q_curve.get_rate(opt.maturity)
        
        model_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *p_ana_params)
        model_mc = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *p_mc_params)
        iv = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, "CALL")
        
        rows.append({
            "T": round(opt.maturity, 3),
            "K": opt.strike,
            "Market": round(opt.market_price, 2),
            "Ana_Price": round(model_ana, 2),
            "Ana_Err": round(model_ana - opt.market_price, 2),
            "MC_Price": round(model_mc, 2),
            "MC_Err": round(model_mc - opt.market_price, 2),
            "IV_Mkt": round(iv, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    print("\n" + "="*80)
    print(f"VALIDATION SAMPLE (First 10) | S0: {S0:.2f} | Yield Used: {q_curve.get_rate(1.0):.4%}")
    print("-" * 80)
    print(df.head(10).to_string(index=False))
    print("-" * 80)
    
    mean_price = df["Market"].mean()
    rmse_ana = np.sqrt((df["Ana_Err"]**2).mean())
    rmse_mc = np.sqrt((df["MC_Err"]**2).mean())
    
    print(f"Analytical RMSE: {rmse_ana:.4f} ({rmse_ana/mean_price:.2%} of avg price)")
    print(f"Monte Carlo RMSE: {rmse_mc:.4f} ({rmse_mc/mean_price:.2%} of avg price)")
    print("="*80)
    print(f"Saved results to: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Risk-Free (r)':<15} | {'Div Yield (q)':<15}")
    print("-" * 60)
    tenors = [(0.0192, "1 Week"), (0.0385, "2 Weeks"), (0.0833, "1 Month"), (0.1667, "2 Months"), (0.25, "3 Months"), (0.3333, "4 Months"), 
              (0.4167, "5 Months"), (0.5, "6 Months"), (0.5833, "7 Months"), (0.6667, "8 Months"), (0.75, "9 Months"), (0.8333, "10 Months"), 
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
    
    # 1. Fetch Treasury Rates
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    
    # 2. Fetch Pillar Data FIRST
    # We need these options to find the market-consistent spot anchor
    raw_df = fetch_raw_data(ticker)
    
    # 3. Derive Implied Spot (Using the 1-month pillar for synchronization)
    S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    
    # 4. Build Dividend Curve relative to that Anchor
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve)
    print_curves(r_curve, q_curve)
    
    # 5. Fetch Broad Options (The 300 options for Heston fit)
    options_raw = fetch_options(ticker, S0_actual, target_size=300)
    
    options_processed = []
    print(f"Processing {len(options_raw)} options (Synthetic Calls)...")
    for opt in options_raw:
        r_T = r_curve.get_rate(opt.maturity)
        q_T = q_curve.get_rate(opt.maturity)
        if opt.option_type == "PUT":
            # Put-Call Parity to create synthetic calls
            price = opt.market_price + (S0_actual * np.exp(-q_T * opt.maturity) - 
                                        opt.strike * np.exp(-r_T * opt.maturity))
            opt.market_price = price
            opt.option_type = "CALL"
        options_processed.append(opt)

    # 6. ANALYTICAL CALIBRATION
    print(f"\n{'='*20} 1. ANALYTICAL CALIBRATION (Albrecher) {'='*20}")
    t0 = time.time()
    calib_analytic = HestonCalibrator(S0=S0_actual, r_curve=r_curve, q_curve=q_curve)
    res_a = calib_analytic.calibrate(options_processed)
    
    print("\n" + "-"*60)
    print(f"ANALYTICAL RESULTS (Time: {time.time()-t0:.2f}s)")
    # Using .get() for safety in case keys vary
    print(f"Obj (Weighted): {res_a.get('weighted_obj', 0):.4f} | RMSE (Price): {res_a.get('rmse', 0):.4f}")
    print(f"k: {res_a['kappa']:.4f} | th: {res_a['theta']:.4f} | xi: {res_a['xi']:.4f} | rho: {res_a['rho']:.4f} | v0: {res_a['v0']:.4f}")
    print("-"*60 + "\n")

    # 7. MONTE CARLO CALIBRATION (Warm Start)
    print(f"{'='*20} 2. MONTE CARLO CALIBRATION (Warm Start) {'='*20}")
    t1 = time.time()
    calib_mc = HestonCalibratorMC(S0=S0_actual, r_curve=r_curve, q_curve=q_curve, 
                                  n_paths=10000, n_steps=400)
    
    # Unpack analytical results to feed into MC
    x0 = [res_a[k] for k in ['kappa','theta','xi','rho','v0']]
    
    # Pass the initial_guess (Ensure your HestonCalibratorMC.calibrate accepts it!)
    res_mc = calib_mc.calibrate(options_processed)

    print("\n" + "-"*60)
    print(f"MONTE CARLO RESULTS (Time: {time.time()-t1:.2f}s)")
    print(f"Obj (Weighted): {res_mc.get('weighted_obj', 0):.4f}")
    print(f"k: {res_mc['kappa']:.4f} | th: {res_mc['theta']:.4f} | xi: {res_mc['xi']:.4f} | rho: {res_mc['rho']:.4f} | v0: {res_mc['v0']:.4f}")
    print("-"*60 + "\n")

    # 8. SAVE & VALIDATE
    save_results(ticker, S0_actual, r_curve, q_curve, res_a, res_mc, options_processed)

if __name__ == "__main__":
    main()