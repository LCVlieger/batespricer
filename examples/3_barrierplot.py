import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from scipy.signal import savgol_filter
from batespricer.market import MarketEnvironment
from batespricer.models.process import BatesProcess
from batespricer.models.mc_pricer import MonteCarloPricer
from batespricer.instruments import BarrierOption, EuropeanOption, BarrierType, OptionType

import matplotlib
matplotlib.use('Agg')


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
        
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f: 
        return json.load(f)
    
def plot_barrier_profile():
    print("--- Generating Ultra-Polished Barrier Profile ---")
    
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
    
    K = env.S0 * 1.05
    B = env.S0 * 0.80
    
    # Generate a slightly denser grid for better smoothing
    spot_range = np.linspace(B * 1.02, env.S0 * 1.15, 35) 
    
    eu_prices, eu_deltas = [], []
    doc_prices, doc_deltas = [], []
    
    pricer = MonteCarloPricer(BatesProcess(env))
    opt_eu = EuropeanOption(K, 1.0, OptionType.CALL)
    opt_doc = BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_OUT, OptionType.CALL)
    
    N_PATHS = 25000 
    N_STEPS = 100

    print("Simulating across Spot range...")
    for i, S in enumerate(spot_range):
        env.S0 = S 
        res_eu = pricer.compute_greeks(opt_eu, n_paths=N_PATHS, n_steps=N_STEPS, seed=42)
        res_doc = pricer.compute_greeks(opt_doc, n_paths=N_PATHS, n_steps=N_STEPS, seed=42)
        
        eu_prices.append(res_eu['price'])
        eu_deltas.append(res_eu['delta'])
        doc_prices.append(res_doc['price'])
        doc_deltas.append(res_doc['delta'])

    print("Applying Quant Desk Polish (Smoothing)...")
    # Savitzky-Golay filter removes MC finite-difference noise while preserving the shape
    window = 11  # Must be odd
    poly = 3     # Cubic polynomial fit
    
    eu_d_smooth = savgol_filter(eu_deltas, window_length=window, polyorder=poly)
    doc_d_smooth = savgol_filter(doc_deltas, window_length=window, polyorder=poly)
    doc_p_smooth = savgol_filter(doc_prices, window_length=window, polyorder=poly)
    eu_p_smooth = savgol_filter(eu_prices, window_length=window, polyorder=poly)

    print("Rendering PDF...")
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left Plot: Prices
    ax1.plot(spot_range, eu_p_smooth, '-', color='#2c3e50', linewidth=2, label='European Call')
    ax1.plot(spot_range, doc_p_smooth, '-', color='#e74c3c', linewidth=2, label='Down-and-Out Call')
    ax1.axvline(x=B, color='black', linestyle='--', linewidth=1.5, label=f'Barrier ($B={B:.1f}$)')
    
    # Shading the price gap (Barrier Discount)
    ax1.fill_between(spot_range, doc_p_smooth, eu_p_smooth, color='grey', alpha=0.15)
    
    ax1.set_title("Price Profile vs. Spot ($S_0$)", fontsize=13)
    ax1.set_xlabel("Spot Price ($S_0$)")
    ax1.set_ylabel("Option Price ($) ")
    ax1.legend(loc='upper left')
    
    # Right Plot: Deltas (The Proof)
    ax2.plot(spot_range, eu_d_smooth, '-', color='#2c3e50', linewidth=2, label='European $\Delta$')
    ax2.plot(spot_range, doc_d_smooth, '-', color='#e74c3c', linewidth=2, label='Down-and-Out $\Delta$')
    ax2.axvline(x=B, color='black', linestyle='--', linewidth=1.5)
    
    # Highlight the exact region you mentioned in your text
    ax2.fill_between(spot_range, eu_d_smooth, doc_d_smooth, 
                     where=(doc_d_smooth > eu_d_smooth), 
                     interpolate=True, color='#e74c3c', alpha=0.2)

    ax2.set_title("Delta Profile vs. Spot ($S_0$)", fontsize=13)
    ax2.set_xlabel("Spot Price ($S_0$)")
    ax2.set_ylabel("Delta ($\Delta$)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('barrier_profile_polished_SPX.pdf', dpi=300)
    print(f"SUCCESS! Plot saved: {os.getcwd()}/barrier_profile_polished.pdf")

if __name__ == "__main__":
    plot_barrier_profile()