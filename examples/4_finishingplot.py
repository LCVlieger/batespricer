import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from copy import deepcopy
from batespricer.market import MarketEnvironment
from batespricer.models.process import BatesProcess
from batespricer.models.mc_pricer import MonteCarloPricer
from batespricer.instruments import BarrierOption, BarrierType, OptionType

import matplotlib
matplotlib.use('Agg')

def load_calibration():
    patterns = ['calibration_*_meta.json', 'results/calibration_*_meta.json', 'examples/calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    if not files: raise FileNotFoundError("No calibration meta file found.")
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f: return json.load(f)

def plot_model_hierarchy():
    print("--- Computing 3-Way Model Risk Hierarchy ---")
    try:
        data = load_calibration()
    except Exception as e:
        print(f"[FATAL] {e}"); return

    p = data.get('analytical', data.get('params', {}))
    m = data.get('market', {})
    
    # 1. BATES: Full systemic risk (Jumps + Stoch Vol)
    env_bates = MarketEnvironment(
        S0=m.get('S0', 6923.54), r=0.041, q=0.0054,
        kappa=p.get('kappa', 1.44), theta=p.get('theta', 0.051),
        xi=p.get('xi', 0.77), rho=p.get('rho', -0.71), v0=p.get('v0', 0.02),
        lamb=p.get('lamb', 0.147), mu_j=p.get('mu_j', -0.19), sigma_j=p.get('sigma_j', 0.21)
    )
    
    # 2. HESTON: 'Equivalent' Diffusion (No jumps, but inflated Vol-of-Vol to fit the smile)
    env_heston = deepcopy(env_bates)
    env_heston.lamb, env_heston.mu_j, env_heston.sigma_j = 0.0, 0.0, 0.0
    env_heston.xi, env_heston.rho = 1.15, -0.92 

    # 3. BLACK-SCHOLES: Baseline (Constant Vol)
    env_bs = deepcopy(env_heston)
    env_bs.kappa, env_bs.xi = 0.0, 0.0 
    
    K = env_bates.S0 * 1.05
    barrier_levels = np.linspace(env_bates.S0 * 0.70, env_bates.S0 * 0.98, 18)
    
    bates_p, heston_p, bs_p = [], [], []
    N_PATHS, N_STEPS = 40000, 252

    print("Simulating Hierarchy (Bates -> Heston -> BS)...")
    for B in barrier_levels:
        opt = BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_OUT, OptionType.CALL)
        
        # Consistent seed for CRN effect
        bates_p.append(MonteCarloPricer(BatesProcess(env_bates)).price(opt, N_PATHS, N_STEPS).price)
        heston_p.append(MonteCarloPricer(BatesProcess(env_heston)).price(opt, N_PATHS, N_STEPS).price)
        bs_p.append(MonteCarloPricer(BatesProcess(env_bs)).price(opt, N_PATHS, N_STEPS).price)
        print(f"  > Barrier {B/env_bates.S0:.1%}: Bates ${bates_p[-1]:.2f} | Heston ${heston_p[-1]:.2f} | BS ${bs_p[-1]:.2f}")

    print("Generating High-Impact Visual...")
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x_axis = barrier_levels / env_bates.S0
    ax1.plot(x_axis, bs_p, 'g--', label='Black-Scholes (No Skew/Jump)', alpha=0.6)
    ax1.plot(x_axis, heston_p, 'o-', color='grey', label='Heston (Diffusion Only)', alpha=0.8)
    ax1.plot(x_axis, bates_p, 's-', color='#c0392b', linewidth=2.5, label='Bates (Full Systemic Risk)')
    
    # Highlight the "Gap Risk" - the area Heston misses
    ax1.fill_between(x_axis, bates_p, heston_p, color='#c0392b', alpha=0.1, label='Jump Risk Discount')
    
    ax1.set_xlabel(r"Barrier Level ($B / S_0$)", fontsize=11)
    ax1.set_ylabel("Option Price ($)", fontsize=11)
    ax1.set_title("Exotic Pricing Hierarchy: Quantifying the Jump Risk Discount", fontsize=14, pad=15)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_hierarchy.pdf', dpi=300)
    print(f"SUCCESS: 'model_hierarchy.pdf' saved.")

if __name__ == "__main__":
    plot_model_hierarchy()