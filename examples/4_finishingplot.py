import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from batespricer.market import MarketEnvironment
from batespricer.models.process import BatesProcess
from batespricer.models.mc_pricer import MonteCarloPricer
from batespricer.instruments import BarrierOption, BarrierType, OptionType

import matplotlib
matplotlib.use('Agg')

def load_calibrations(ticker):
    """Loads Bates from original analytic run, and Heston/BS from recalibrated run."""
    clean_ticker = ticker.replace('^', '')
    
    bates_patterns = [
        f'**/calibration_Analytic_{ticker}_*_meta.json', 
        f'**/calibration_Analytic_{clean_ticker}_*_meta.json',
        'calibration_Analytic_*_meta.json'
    ]
    bates_files = []
    for p in bates_patterns: bates_files.extend(glob.glob(p, recursive=True))
    if not bates_files: raise FileNotFoundError(f"Bates calibration for {ticker} not found.")
    latest_bates_file = max(bates_files, key=os.path.getctime)
    
    recal_patterns = [
        f'**/recalibrated_models_{clean_ticker}_meta.json',
        f'**/recalibrated_models_{ticker}_meta.json',
        '**/recalibrated_models.json'
    ]
    recal_files = []
    for p in recal_patterns: recal_files.extend(glob.glob(p, recursive=True))
    if not recal_files: raise FileNotFoundError(f"Recalibrated models file not found.")
    latest_recal_file = max(recal_files, key=os.path.getctime)

    with open(latest_bates_file, 'r') as f: bates_data = json.load(f)
    with open(latest_recal_file, 'r') as f: recal_data = json.load(f)

    return bates_data, recal_data

def get_plot_data(ticker):
    """Calculates the pricing data for a given ticker and returns it."""
    try:
        bates_data, recal_data = load_calibrations(ticker)
    except Exception as e:
        print(f"[ERROR] {e}"); return None

    m = bates_data['market']
    S0_orig = m['S0']
    r_val = m['r_sample'].get('0.0833Y', 0.04)
    q_val = m['q_sample'].get('0.0833Y', 0.005)

    p_bates = bates_data['analytical']
    p_heston = recal_data['models']['Heston']
    p_bs = recal_data['models']['BS']

    T_fixed, K_fixed, B_fixed = 0.0833, S0_orig * 1.02, S0_orig * 0.92
    spot_range = np.linspace(B_fixed * 1.005, S0_orig * 1.045, 18)

    bates_p, heston_p, bs_p = [], [], []
    N_PATHS, N_STEPS = 250000, 252

    for S in spot_range:
        m_env = {
            'Bates': MarketEnvironment(S, r_val, q_val, **{k: p_bates[k] for k in ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']}),
            'Heston': MarketEnvironment(S, r_val, q_val, **{k: p_heston[k] for k in ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']}),
            'BS': MarketEnvironment(S, r_val, q_val, **{k: p_bs[k] for k in ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']})
        }
        opt = BarrierOption(K_fixed, T_fixed, B_fixed, BarrierType.DOWN_AND_OUT, OptionType.CALL)
        
        bates_p.append(MonteCarloPricer(BatesProcess(m_env['Bates'])).price(opt, N_PATHS, N_STEPS).price / S)
        heston_p.append(MonteCarloPricer(BatesProcess(m_env['Heston'])).price(opt, N_PATHS, N_STEPS).price / S)
        bs_p.append(MonteCarloPricer(BatesProcess(m_env['BS'])).price(opt, N_PATHS, N_STEPS).price / S)
        print(bates_p)
        print(heston_p)
        print(bs_p)
        
    return {
        'ticker': ticker,
        'spot_range': spot_range,
        'bs_p': bs_p,
        'heston_p': heston_p,
        'bates_p': bates_p,
        'B_fixed': B_fixed
    }

def plot_single_hierarchy(data, y_lim, out_name):
    """Plots a single hierarchy with a forced y_lim."""
    C_BATES, C_HESTON, C_BS = "#000F3B", "#1D408B", "#367BDC"
    PANE_GRAY, GRID_STYLE = (0.95, 0.95, 0.95, 1.0), (0.68, 0.68, 0.68, 0.5)

    fig, ax1 = plt.subplots(figsize=(10, 7.25), facecolor='white')
    ax1.set_facecolor(PANE_GRAY)
    
    ax1.plot(data['spot_range'], data['bs_p'], '--', color=C_BS, label='Black-Scholes', alpha=0.7, linewidth=3.0)
    ax1.plot(data['spot_range'], data['heston_p'], 'o-', color=C_HESTON, label='Heston', alpha=0.9, linewidth=3.0, markersize=6)
    ax1.plot(data['spot_range'], data['bates_p'], 's-', color=C_BATES, label='Bates', linewidth=4.0, markersize=6)
    
    ax1.axvline(x=data['B_fixed'], color='black', linestyle='--', linewidth=2.5, label='Knock-Out Barrier')
    ax1.fill_between(data['spot_range'], data['bates_p'], data['heston_p'], color=C_BATES, alpha=0.15)

    ax1.set_yscale('log')
    ax1.set_ylim(y_lim) # Forces the shared limit
    
    ax1.grid(True, which="major", axis='both', linewidth=1.0, color=GRID_STYLE)
    ax1.grid(False, which="minor")
    
    for s in ['top', 'right']: ax1.spines[s].set_visible(False)
    for s in ['left', 'bottom']: 
        ax1.spines[s].set_color('black')
        ax1.spines[s].set_linewidth(1.5)

    ax1.set_xlabel("Market Spot Price ($S_0$)", color="black", fontsize=18, labelpad=12)
    ax1.set_ylabel("Relative Price ($c/S_0$)", color="black", fontsize=18, labelpad=12)
    
    ax1.tick_params(axis='both', which='major', colors='black', labelsize=16, width=1.5, length=6)
    
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc='lower right', frameon=True, 
               facecolor=PANE_GRAY, edgecolor='none', labelcolor="black", 
               framealpha=0.95, fontsize=16)

    plt.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.16)
    plt.savefig(out_name, format='pdf', bbox_inches='tight', pad_inches=0.01, dpi=800)
    print(f"SUCCESS: '{out_name}' saved.")

if __name__ == "__main__":
    # 1. Gather data for both
    spx_data = get_plot_data("^SPX")
    aapl_data = get_plot_data("AAPL")

    # 2. Find Global Y-Limits
    all_prices = spx_data['bs_p'] + spx_data['bates_p'] + aapl_data['bs_p'] + aapl_data['bates_p'] + spx_data['heston_p'] + aapl_data['heston_p']
    global_min = min(all_prices) * 0.6
    global_max = max(all_prices) * 2.0
    shared_ylim = (global_min, global_max)

    # 3. Generate individual plots with shared limit
    plot_single_hierarchy(spx_data, shared_ylim, 'model_hierarchy_spot_SPX.pdf')
    plot_single_hierarchy(aapl_data, shared_ylim, 'model_hierarchy_spot_AAPL.pdf')