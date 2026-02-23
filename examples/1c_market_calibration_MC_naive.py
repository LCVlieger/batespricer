import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from batespricer.calibration import BatesCalibratorMC 
from batespricer.analytics import implied_volatility, BatesAnalyticalPricer

class InterpolatedCurve:
    def __init__(self, curve_dict):
        times = sorted([float(k.replace('Y', '')) for k in curve_dict.keys()])
        rates = [curve_dict[f"{t:.4f}Y"] for t in times]
        self.interp = interp1d(times, rates, kind='linear', fill_value=(rates[0], rates[-1]), bounds_error=False)

    def get_rate(self, T):
        return float(self.interp(T))

class SimpleOption:
    def __init__(self, T, K, type_str, price, spread):
        self.maturity = T
        self.strike = K
        self.option_type = type_str
        self.market_price = price
        self.ask = price + (spread / 2.0)
        self.bid = price - (spread / 2.0)

def save_results(ticker, S0, r_curve, q_curve, res_params, options):
    os.makedirs("results", exist_ok=True)
    base_name = f"results/calibration_MC_{ticker}_20260208_022951"
    
    tenors = [(0.02, "1w"), (0.04, "2w"), (0.0833, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, _ in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, _ in tenors}
    
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    p_values = [res_params[k] for k in keys] if isinstance(res_params, dict) else list(res_params.x)
    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p_values
    
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    types = np.array([o.option_type for o in options])
    r_T = np.array([r_curve.get_rate(o.maturity) for o in options])
    q_T = np.array([q_curve.get_rate(o.maturity) for o in options])
    
    model_prices = BatesAnalyticalPricer.price_vectorized(
        S0, strikes, mats, r_T, q_T, types, 
        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
    )
    
    rows = []
    w_sq_err = []
    norm_w = np.ones(len(options))
    
    for i, opt in enumerate(options):
        r_val = r_T[i]
        q_val = q_T[i]
        model_p = model_prices[i]
        err = model_p - opt.market_price
        
        w_sq_err.append((err * norm_w[i])**2)
        
        try:
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r_val, q_val, opt.option_type)
            iv_model = implied_volatility(model_p, S0, opt.strike, opt.maturity, r_val, q_val, opt.option_type)
        except:
            iv_mkt = 0.0
            iv_model = 0.0
            
        rows.append({
            "T": round(opt.maturity, 3), 
            "K": opt.strike, 
            "Spread": opt.ask - opt.bid,
            "Weight": 1 / max(abs(opt.ask - opt.bid), 0.01), 
            "Type": opt.option_type, 
            "Market": round(opt.market_price, 2), 
            "Model": round(model_p, 2), 
            "Err": round(err, 2), 
            "IV_M": round(iv_mkt, 4), 
            "IV_P": round(iv_model, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    rmse = float(np.sqrt((df["Err"]**2).mean()))
    res_dict = dict(zip(keys, p_values))
    res_dict['rmse'] = rmse
    
    meta = {
        "model": "Bates_MC_Val_Analytic", 
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample}, 
        "params": res_dict
    }
    
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    print("="*80 + f"\nVALIDATION | S0: {S0:.2f}\n" + "-"*80)
    print(df.head(10).to_string(index=False))
    print("-" * 80)
    print(f"Analytic RMSE: {rmse:.4f} | Weighted Obj: {np.sqrt(np.mean(w_sq_err)):.4f}")
    print(f"Saved: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60 + f"\n{'Tenor':<10} | {'Rate (r)':<15} | {'Yield (q)':<15}\n" + "-"*60)
    tenors = [(0.02, "1w"), (0.04, "2w"), (0.0833, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y")]
    for t, label in tenors:
        print(f"{label:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")

    print("="*60 + "\n")

def main():
    json_path = "results/calibration_Analytic_^SPX_20260208_022951_meta.json"
    csv_path = "results/calibration_Analytic_^SPX_20260208_022951_prices.csv"
    ticker = "SPX"
    
    with open(json_path, 'r') as f:
        meta = json.load(f)
        
    S0 = meta['market']['S0']
    r_curve = InterpolatedCurve(meta['market']['r_sample'])
    q_curve = InterpolatedCurve(meta['market']['q_sample'])
    
    print_curves(r_curve, q_curve)
    
    df_prices = pd.read_csv(csv_path)
    options_processed = []
    
    for _, row in df_prices.iterrows():
        options_processed.append(SimpleOption(
            row['T'], row['K'], row['Type'], row['Market'], row['Spread']
        ))

    mc_calib = BatesCalibratorMC(S0=S0, r_curve=r_curve, q_curve=q_curve, n_paths=5000, n_steps=5000)
    mc_calib._precompute(options_processed)
    
    short_opts = [o for o in options_processed if o.maturity < 0.1]
    closest = min(short_opts, key=lambda x: abs(x.strike - S0))
    
    iv = implied_volatility(
        closest.market_price, S0, closest.strike, closest.maturity, 
        r_curve.get_rate(closest.maturity), q_curve.get_rate(closest.maturity), closest.option_type
    )
    print(f"LOCK v0: {iv**2:.4f} (Vol: {iv:.1%})")
    
    def obj(p):
        try:
            model_p, market_p, weights = mc_calib.get_prices(p)
            return np.sqrt(np.mean(((model_p - market_p) * weights)**2))
        except:
            return 1e10
            
    def cb(xk):
        print(f" [MC-Iter] Obj: {obj(xk):10.4f} | v0:{xk[4]:.4f} th:{xk[1]:.4f} ka:{xk[0]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f}")
        
    t0 = time.time()
    x0 = [1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]
    bounds = [
        (1.0, 5.0), (0.001, 0.5), (0.01, 0.9), (-0.99, 0.0), 
        (0.001, 0.1), (0.0, 0.5), (-0.3, 0.0), (0.05, 0.3)
    ]
    
    res = minimize(
        obj, x0, method='SLSQP', bounds=bounds, callback=cb, 
        tol=1e-8, options={'maxiter': 500, 'eps': 3e-3}
    )
    
    print(f"DONE ({time.time()-t0:.2f}s)")
    
    final_params = dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x))
    save_results(ticker, S0, r_curve, q_curve, final_params, options_processed)

if __name__ == "__main__": 
    main()