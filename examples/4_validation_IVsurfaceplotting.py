import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter 

# Local package imports
try:
    from heston_pricer.calibration import implied_volatility, HestonCalibrator, SimpleYieldCurve
    from heston_pricer.analytics import HestonAnalyticalPricer
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from heston_pricer.calibration import implied_volatility, HestonCalibrator, SimpleYieldCurve
    from heston_pricer.analytics import HestonAnalyticalPricer

class ReconstructedOption:
    def __init__(self, strike, maturity, price, option_type="CALL"):
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.market_price = float(price)
        self.option_type = str(option_type)

def load_latest_calibration():
    patterns = ['results/calibration_*_meta.json', 'calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    
    if not files: raise FileNotFoundError("No calibration meta file found.")
    
    latest_meta = max(files, key=os.path.getctime)
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading Artifact: {base_name}...")
    
    with open(latest_meta, 'r') as f: data = json.load(f)
    
    def reconstruct_curve(curve_data):
        if isinstance(curve_data, dict) and 'tenors' in curve_data:
            return SimpleYieldCurve(curve_data['tenors'], curve_data['rates'])
        else:
            val = float(curve_data)
            return SimpleYieldCurve([0.0, 30.0], [val, val])

    # Reconstruct both curves to avoid float-dict operand errors
    r_curve = reconstruct_curve(data['market']['r'])
    q_curve = reconstruct_curve(data['market']['q'])

    csv_file = f"{base_name}_prices.csv"
    market_options = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            otype = row['Type'] if 'Type' in row else "CALL"
            market_options.append(ReconstructedOption(row['K'], row['T'], row['Mkt'], otype))

    return data, r_curve, q_curve, market_options, base_name

def select_best_parameters(data):
    res_ana = data.get('analytical', {})
    res_mc = data.get('monte_carlo', data.get('monte_carlo_results', {}))

    def get_score(res):
        if not res or 'fun' not in res: return float('inf')
        return res['fun']

    score_ana = get_score(res_ana)
    score_mc = get_score(res_mc)

    if score_mc < score_ana:
        print(f"\n[Selection] Monte Carlo Win (Err: {score_mc:.4f} < Ana: {score_ana:.4f})")
        return res_mc, "Monte Carlo"
    elif score_ana < float('inf'):
        print(f"\n[Selection] Analytical Win (Err: {score_ana:.4f} < MC: {score_mc:.4f})")
        return res_ana, "Analytical"
    else:
        print("\n[Selection] No valid results found. Using default guess.")
        return {'kappa':2.0, 'theta':0.04, 'xi':0.5, 'rho':-0.7, 'v0':0.04}, "Default"

def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']

    # --- 1. CONFIGURATION ---
    LOWER_M, UPPER_M = 0.5, 1.8 
    LOWER_T, UPPER_T = 0.1, 2.5
    GRID_DENSITY = 100 

    M_range = np.linspace(LOWER_M, UPPER_M, GRID_DENSITY)
    T_range = np.linspace(LOWER_T, UPPER_T, GRID_DENSITY)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- 2. CALCULATION ---
    print(f"-> Generating Surface for: kappa={kappa:.2f}, xi={xi:.2f}, v0={v0:.3f}")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            
            # Extract scalars from yield curves for each grid point
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)

            price = HestonAnalyticalPricer.price_european_call(
                S0, S0 * M_val, T_val, r_T, q_T, kappa, theta, xi, rho, v0
            )
            try:
                iv = implied_volatility(price, S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                Z[i, j] = np.nan

    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1).values
    Z_smooth = gaussian_filter(Z, sigma=0.8)

    # --- 3. PLOTTING (Exact Aesthetic Match) ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z_smooth, cmap=cm.RdYlBu_r, 
                               rcount=100, ccount=100,  
                               edgecolor='black', linewidth=0.085, alpha=0.8,                      
                               shade=False, antialiased=True, zorder=1)

        if market_options:
            plot_opts = [
                o for o in market_options 
                if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)
            ]
            
            valid_needles = 0
            for opt in plot_opts:
                m_mkt, t_mkt = opt.strike / S0, opt.maturity
                try:
                    r_T_mkt = r_curve.get_rate(t_mkt)
                    q_T_mkt = q_curve.get_rate(t_mkt)
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)
                    if iv_mkt < 0.01 or iv_mkt > 2.5: continue
                except: continue

                m_idx = (np.abs(M_range - m_mkt)).argmin()
                t_idx = (np.abs(T_range - t_mkt)).argmin()
                iv_mod = Z_smooth[t_idx, m_idx]

                if np.isnan(iv_mod): continue
                
                valid_needles += 1
                is_above = iv_mkt >= iv_mod
                dot_zorder = 10 if is_above else 1

                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod, iv_mkt], 
                        color='white', linestyle='-', linewidth=0.8, alpha=0.65, zorder=dot_zorder)
                
                lbl = 'Market Price-IV' if valid_needles == 1 else ""
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mkt], 
                        marker='o', linestyle='None',
                        color="#F0F0F0", markersize=4.0, alpha=0.9, 
                        zorder=dot_zorder, label=lbl)

        # --- 4. AESTHETICS ---
        ax.dist = 11
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) 
        ax.set_zlim(0.35, 0.65) 
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
        
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.view_init(elev=28, azim=-115) 

        fig.text(0.535, 0.84, rf"Heston Implied Volatility Surface: {ticker}", 
                 color='white', fontsize=16, fontweight='bold', family='monospace', ha='center')
        subtitle = rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$"
        fig.text(0.535, 0.81, subtitle, color='#AAAAAA', fontsize=10, family='monospace', ha='center')

        # --- 5. PERFORMANCE METRICS ---
        comparison_text = (
            f"Model Source: {source_name}\n"
            f"-------------------\n"
            f"Final RMSE (IV):    {params.get('rmse_iv', 0):.5f}\n"
            f"Obj Function:       {params.get('fun', 0):.5f}\n"
            f"Feller Condition:   {'Met' if (2*kappa*theta > xi**2) else 'Violated'}\n"
            f"Outliers Removed:   {dropped_count}"
        )
        print("\n" + comparison_text)
        
        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=10)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=10)
        ax.set_zlabel(r'Implied Volatility (%)', color='white', labelpad=10)

        if market_options and valid_needles > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.157, 0.797), frameon=False, labelcolor="#D7D7D7", fontsize=10)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.outline.set_visible(False)

        save_path = f"{filename}_surface_refined.png"
        plt.savefig(save_path, dpi=600, facecolor='black', bbox_inches='tight')
        print(f"-> Saved: {save_path}")
        plt.close()

def main():
    try:
        data, r_curve, q_curve, market_options, base_name = load_latest_calibration()
        S0 = data['market']['S0']
        
        best_params, source_name = select_best_parameters(data)
        ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
        
        print(f"\n[Direct Plot] Using {source_name} parameters directly (skipping refinement).")
        
        plot_surface_professional(
            S0, r_curve, q_curve, best_params, ticker, base_name, 
            market_options, data, 0, source_name
        )
        
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()