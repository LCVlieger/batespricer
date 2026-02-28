import json
import glob
import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from batespricer.calibration import BatesCalibrator, BatesCalibratorFast

class RobustYieldCurve:
    def __init__(self, curve_data):
        times, rates = [], []
        
        if isinstance(curve_data, dict):
            for k, v in curve_data.items():
                try:
                    t_str = str(k).lower().replace("y", "").replace("week", "")
                    times.append(float(t_str))
                    rates.append(float(v))
                except: continue
        elif hasattr(curve_data, 'tenors'):
            times, rates = curve_data.tenors, curve_data.rates
        else:
            times, rates = [0.0, 30.0], [float(curve_data), float(curve_data)]

        sorted_pairs = sorted(zip(times, rates))
        self.ts = np.array([p[0] for p in sorted_pairs])
        self.rs = np.array([p[1] for p in sorted_pairs])
        
        self.interp = interp1d(self.ts, self.rs, kind='linear', 
                               bounds_error=False, fill_value=(self.rs[0], self.rs[-1]))

    def get_rate(self, T):
        return float(self.interp(max(T, 1e-4)))
class ReconstructedOption:
    def __init__(self, strike, maturity, price, bid, ask, option_type="CALL"):
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.market_price = float(price)
        self.bid = float(bid)
        self.ask = float(ask)
        self.option_type = str(option_type).upper()
def load_latest_calibration():
    patterns = ['results/calibration_*_meta.json', 'calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    
    if not files: raise FileNotFoundError("No calibration meta file found.")
    
    # Grab the absolute latest file
    latest_meta = sorted(files, key=os.path.getctime)[-1] 
    base_name = latest_meta.replace("_meta.json", "")
    
    with open(latest_meta, 'r') as f: 
        data = json.load(f)
    
    # CRITICAL: Lock the curves to the exact samples saved in the meta file
    r_curve = RobustYieldCurve(data['market'].get('r', data['market'].get('r_sample')))
    q_curve = RobustYieldCurve(data['market'].get('q', data['market'].get('q_sample')))

    market_options = []
    csv_file = f"{base_name}_prices.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Mapping to your exact CSV headers: 'K', 'T', 'Market', 'Bid', 'Ask', 'Type'
        for _, row in df.iterrows():
            market_options.append(ReconstructedOption(
                strike=row['K'],
                maturity=row['T'],
                price=row['Market'],
                bid=row['Bid'],
                ask=row['Ask'],
                option_type=row['Type']
            ))

    return data, r_curve, q_curve, market_options, base_name
def main():
    print("="*60)
    print("--- Starting Multi-Model Recalibration ---")
    try:
        data, r_curve, q_curve, market_options, base_name = load_latest_calibration()
        S0 = data['market'].get('S0', 6923.54)
        print(f"Loaded {len(market_options)} options from: {base_name}")
    except Exception as e:
        print(f"[FATAL] Could not load calibration data: {e}")
        return

    # Initialize the updated calibrator
    calibrator = BatesCalibratorFast(S0=S0, r_curve=r_curve, q_curve=q_curve)
    
    models_to_run = ["Bates", "Heston", "BS"]
    results_dict = {}

    for model_name in models_to_run:
        print(f"\nCalibrating {model_name}...")
        t0 = time.time()
        
        # Call the new calibrate method with the model argument
        res = calibrator.calibrate(market_options, sigma_cap=2.0, model=model_name)
        
        elapsed = time.time() - t0
        print(f"[{model_name}] Time: {elapsed:.2f}s | RMSE: {res.get('rmse', 0):.4f}")
        
        # Print parameters cleanly
        for k in ['v0', 'kappa', 'theta', 'xi', 'rho', 'lamb', 'mu_j', 'sigma_j']:
            print(f"  {k:<8}: {res[k]:.4f}")
            
        results_dict[model_name] = res

    # Save the combined results for the plotting script
    output_file = "recalibrated_models.json"
    with open(output_file, "w") as f:
        json.dump({
            "market": {"S0": S0, "r_sample": data['market'].get('r_sample'), "q_sample": data['market'].get('q_sample')},
            "models": results_dict
        }, f, indent=4)
        
    print("="*60)
    print(f"SUCCESS! All models calibrated and saved to '{output_file}'.")

if __name__ == "__main__":
    main()