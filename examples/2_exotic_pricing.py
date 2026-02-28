import json
import glob
import os
import numpy as np
import pandas as pd
from batespricer.market import MarketEnvironment
from batespricer.models.process import BatesProcess
from batespricer.models.mc_pricer import MonteCarloPricer
from batespricer.instruments import BarrierOption, BarrierType, AsianOption, EuropeanOption, OptionType

def load_calibration():
    patterns = [
        'calibration_*_meta.json', 
        'examples/calibration_*_meta.json', 
        '../calibration_*_meta.json',
        'results/calibration_*_meta.json'
    ]
    
    files = []
    for p in patterns: 
        files.extend(glob.glob(p))
        
    if not files: 
        raise FileNotFoundError("No calibration meta file found.")
        
    latest_file = min(files, key=os.path.getctime)
    with open(latest_file, 'r') as f: 
        return json.load(f)

def compute(pricer, option, name):
    result = pricer.compute_greeks(option, n_paths=100_000, n_steps=1000,seed=42)
    
    return {
        "Product": name, 
        "Price": result['price'], 
        "Delta": result['delta'], 
        "Gamma": result['gamma'], 
        "Vega": result['vega_v0']
    }

def main():
    try:
        data = load_calibration()
    except Exception as e:
        print(f"[FATAL] {e}")
        return

    # Robust extraction based on the latest calibration JSON structure
    p = data.get('analytical', data.get('params', data.get('monte_carlo_results', {})))
    m = data.get('market', {})
    
    r_data = m.get('r_sample', m.get('r', 0.05))
    q_data = m.get('q_sample', m.get('q', 0.0))
    
    r_val = r_data.get('1.0000Y', 0.05) if isinstance(r_data, dict) else float(r_data)
    q_val = q_data.get('1.0000Y', 0.0) if isinstance(q_data, dict) else float(q_data)

    env = MarketEnvironment(
        S0=m.get('S0', 100.0), 
        r=r_val, 
        q=q_val, 
        kappa=p.get('kappa', 1.0), 
        theta=p.get('theta', 0.04), 
        xi=p.get('xi', 0.1), 
        rho=p.get('rho', -0.7), 
        v0=p.get('v0', 0.04),
        lamb=p.get('lamb', 0.0), 
        mu_j=p.get('mu_j', 0.0), 
        sigma_j=p.get('sigma_j', 0.0)
    )
    
    pricer = MonteCarloPricer(BatesProcess(env))
    
    S0 = env.S0
    K = S0 * 1.05
    B = S0 * 0.80

    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Pricing Exotics via Bates Model (S0={S0:.2f})...")
    
    results = [
        compute(pricer, EuropeanOption(K, 1.0, OptionType.CALL), "European Call"),
        compute(pricer, BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_OUT, OptionType.CALL), "Down-Out Call"),
        compute(pricer, BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_IN, OptionType.CALL), "Down-In Call"),
        compute(pricer, AsianOption(K, 1.0, OptionType.CALL), "Asian Call")
    ]
    
    df = pd.DataFrame(results)
    print(df.set_index("Product").to_string(float_format="{:.4f}".format))

if __name__ == "__main__":
    main()