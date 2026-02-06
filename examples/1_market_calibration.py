import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from heston_pricer.calibration import BatesCalibrator 
from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
from heston_pricer.data import (
    fetch_treasury_rates_fred, 
    fetch_raw_data, 
    fetch_options, 
    get_market_implied_spot, 
    ImpliedDividendCurve
)

# =================================================================
# 3. SAVING & VALIDATION 
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_params, options):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_Analytic_{ticker}_{timestamp}"
    
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, label in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, label in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments using Final MC Parameters...")

    # 1. Weights for RMSE calculation (1.0 to match your analytical file)
    raw_weights = [1.0 for _ in options]
    normalized_weights = np.array(raw_weights) / np.mean(raw_weights)

    # 2. Extract Parameter Values
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    if isinstance(res_params, dict):
        p_values = [res_params[k] for k in keys]
    else:
        p_values = list(res_params.x)
    print(p_values)
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    types = np.array([o.option_type for o in options])
    r_T = np.array([r_curve.get_rate(o.maturity) for o in options])
    q_T = np.array([q_curve.get_rate(o.maturity) for o in options])
    theta, xi, kappa, rho, v0, lamb, mu_j, sigma_j = p_values
    # 3. Final High-Precision Pass
    model_prices = BatesAnalyticalPricer.price_vectorized(S0, strikes, mats, r_T, q_T, types,  theta, xi, kappa, rho, v0, lamb, mu_j, sigma_j )

    rows = []
    weighted_sq_err = []
    
   
    r_vec = np.array([r_curve.get_rate(t) for t in mats])
    q_vec = np.array([q_curve.get_rate(t) for t in mats])

    for i, opt in enumerate(options):
        r, q = r_vec[i], q_vec[i]
        model_p = model_prices[i]
        err = model_p - opt.market_price
        weighted_sq_err.append((err * normalized_weights[i])**2)
        
        try:
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, opt.option_type)
            iv_model = implied_volatility(model_p, S0, opt.strike, opt.maturity, r, q, opt.option_type)
        except:
            iv_mkt, iv_model = 0.0, 0.0
        
        rows.append({
            "T": round(opt.maturity, 3), "K": opt.strike, "Type": opt.option_type,
            "Market": round(opt.market_price, 2), "Model": round(model_p, 2), 
            "Err": round(err, 2), "IV_Mkt": round(iv_mkt, 4), "IV_Model": round(iv_model, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    # --- METRICS CALCULATION ---
    rmse_val = float(np.sqrt((df["Err"]**2).mean()))
    weighted_obj_val = float(np.sqrt(np.mean(weighted_sq_err)))

    # --- JSON CONSTRUCTION (Injected with scoring keys) ---
    # We use 'analytical' as the key so the plotter can find it easily
    res_dict = dict(zip(keys, p_values))
    res_dict['weighted_obj'] = weighted_obj_val
    res_dict['rmse'] = rmse_val

    meta = {
        "model": "Bates_MC",
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample},
        "analytical": res_dict, # Using 'analytical' key for plotter compatibility
        "notes": "Calibrated directly via Monte Carlo"
    }
    
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    
    print("\n" + "="*80)
    print(f"VALIDATION SAMPLE (First 10) | S0: {S0:.2f}")
    print("-" * 80)
    print(df.head(10).to_string(index=False))
    print("-" * 80)
    print(f"Final MC RMSE: {rmse_val:.4f} | Weighted Obj: {weighted_obj_val:.4f}")
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
    ticker = "SPY" #"^SPX"
    
    print(f"Initializing Bates Calibration for {ticker}...")
    
    # 1. Data Fetching
    # r_curve now automatically adds the BOX_SPREAD in data.py
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    raw_df = fetch_raw_data(ticker)
    
    # S0_actual uses the higher r_curve to synchronize correctly
    S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve, ticker)
    print_curves(r_curve, q_curve)
    
    # options_processed no longer contains the .forward field
    options_processed = fetch_options(ticker, S0_actual, target_size=150)
    print(f"Processing {len(options_processed)} standard OTM options...")
    
    # 3. Calibration
    print(f"\n{'='*20} BATES ANALYTICAL CALIBRATION {'='*20}")
    t0 = time.time()
    
    calib_analytic = BatesCalibrator(S0=S0_actual, r_curve=r_curve, q_curve=q_curve)
    res_a = calib_analytic.calibrate(options_processed)
    
    print(f"CALIBRATION RESULTS (Time: {time.time()-t0:.2f}s)")
    print(f"Obj (Weighted): {res_a.get('weighted_obj', 0):.4f} | RMSE (Price): {res_a.get('rmse', 0):.4f}")
    
    # Detailed Parameter Output
    print("-" * 60)
    print(f"Heston Component:")
    print(f"  v0:      {res_a['v0']:.4f}")
    print(f"  kappa:   {res_a['kappa']:.4f}")
    print(f"  theta:   {res_a['theta']:.4f}")
    print(f"  xi:      {res_a['xi']:.4f}")
    print(f"  rho:     {res_a['rho']:.4f}")
    print(f"Jump Component (Merton):")
    print(f"  lambda:   {res_a['lamb']:.4f}  (Jumps per year)")
    print(f"  mu_J:     {res_a['mu_j']:.4f}  (Mean log-jump size)")
    print(f"  sigma_J:  {res_a['sigma_j']:.4f}  (Jump size volatility)")
    print("-" * 60)

    # 4. Saving
    save_results(ticker, S0_actual, r_curve, q_curve, res_a, options_processed)

if __name__ == "__main__":
    main()