import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter 
from scipy.interpolate import interp1d

# Local package imports
try:
    # UPDATED IMPORT: BatesAnalyticalPricer
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass

# --- 1. COMPATIBILITY HELPERS ---
class RobustYieldCurve:
    """
    Parses the new dictionary-style curves {"0.08Y": 0.035, ...}
    and allows extrapolation for the T=2.5 plotting range.
    """
    def __init__(self, curve_data):
        times, rates = [], []
        
        # Handle Dictionary Format (New)
        if isinstance(curve_data, dict):
            for k, v in curve_data.items():
                try:
                    # Strip 'Y', 'week', etc to get float
                    t_str = str(k).lower().replace("y", "").replace("week", "")
                    times.append(float(t_str))
                    rates.append(float(v))
                except: continue
        # Handle List/Object Format (Old)
        elif hasattr(curve_data, 'tenors'):
            times = curve_data.tenors
            rates = curve_data.rates
        else:
            # Scalar fallback
            times = [0.0, 30.0]
            rates = [float(curve_data), float(curve_data)]

        # Sort for interpolation
        sorted_pairs = sorted(zip(times, rates))
        self.ts = np.array([p[0] for p in sorted_pairs])
        self.rs = np.array([p[1] for p in sorted_pairs])
        
        # Linear interp with Flat Extrapolation (so T=2.5 works)
        self.interp = interp1d(self.ts, self.rs, kind='linear', 
                               bounds_error=False, 
                               fill_value=(self.rs[0], self.rs[-1]))

    def get_rate(self, T):
        return float(self.interp(max(T, 1e-4)))

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
    files_sorted = sorted(files, key=os.path.getctime)
    latest_meta = files_sorted[2] # max(files, key=os.path.getctime)
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading Artifact: {base_name}...")
    
    with open(latest_meta, 'r') as f: data = json.load(f)
    
    # Use RobustYieldCurve to handle the dictionary format
    r_data = data['market'].get('r_sample', data['market'].get('r'))
    q_data = data['market'].get('q_sample', data['market'].get('q'))
    
    r_curve = RobustYieldCurve(r_data)
    q_curve = RobustYieldCurve(q_data)

    csv_file = f"{base_name}_prices.csv"
    market_options = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            # Handle variable column names
            k = row.get('K', row.get('Strike', 0))
            t = row.get('T', row.get('Maturity', 0))
            p = row.get('Market', row.get('Price', 0))
            otype = row.get('Type', "CALL")
            market_options.append(ReconstructedOption(k, t, p, otype))

    return data, r_curve, q_curve, market_options, base_name

def select_best_parameters(data):
    res_mc = data.get('analytical', {})
    print(data)
    print(res_mc)
    # Support both old 'fun' and new 'weighted_obj
    return res_mc, "Monte Carlo"
    

# --- 2. EXACT PLOTTING FUNCTION (Updated for Bates) ---
def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    # Extract Heston Params
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    
    # Extract Bates Params (Default to 0.0 for pure Heston compatibility)
    lamb = params.get('lamb', 0.0)
    mu_j = params.get('mu_j', 0.0)
    sigma_j = params.get('sigma_j', 0.0)
    
    is_bates = lamb > 0.0

    # --- 1. CONFIGURATION ---
    LOWER_M, UPPER_M = 0.685, 1.315 #0.685, 1.315  #0.7, 1.3 #0.685, 1.315 
    LOWER_T, UPPER_T = 0.04, 1.5 
    GRID_DENSITY = 100 # 100 is smoother but slower

    M_range = np.linspace(LOWER_M, UPPER_M, GRID_DENSITY)
    T_range = np.linspace(LOWER_T, UPPER_T, GRID_DENSITY)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- 2. CALCULATION ---
    print(f"-> Generating Surface for: {ticker}")
    print(f"   Model: {'Bates (Heston + Jumps)' if is_bates else 'Heston'}")
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            
            # Extract scalars from yield curves for each grid point
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)
            
            # --- CALL BATES PRICER ---
            # Note: We pass scalars wrapped in arrays if the pricer expects arrays, 
            # or just rely on the vectorized function handling scalars (it returns an array).
            # BatesAnalyticalPricer.price_european_call_vectorized returns array[float]
            
            prices = BatesAnalyticalPricer.price_european_call_vectorized(
                S0, np.array([S0 * M_val]), np.array([T_val]), np.array([r_T]), np.array([q_T]),
                kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
            )
            price = float(prices[0])

            try:
                iv = implied_volatility(price, S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                Z[i, j] = np.nan

    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1).values
    Z_smooth = gaussian_filter(Z, sigma=0)

    # --- 3. PLOTTING ---
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
                    
                    # 1. MARKET IV
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)
                    
                    # 2. EXACT MODEL IV (Bates)
                    prices_mod = BatesAnalyticalPricer.price_vectorized(
                        S0, np.array([opt.strike]), np.array([t_mkt]), np.array([r_T_mkt]), np.array([q_T_mkt]), np.array([opt.option_type]),
                        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                    )
                    price_mod = float(prices_mod[0])
                    
                    iv_mod_exact = implied_volatility(price_mod, S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)

                    if iv_mkt < 0.01 or iv_mkt > 2.5: continue
                except: 
                    continue

                valid_needles += 1
                is_above = iv_mkt >= iv_mod_exact
                dot_zorder = 10 if is_above else 1

                # DRAW NEEDLE
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod_exact, iv_mkt], 
                        color='white', linestyle='-', linewidth=0.8, alpha=0.65, zorder=dot_zorder)
                
                # DRAW DOT
                lbl = 'Market Price-IV' if valid_needles == 1 else ""
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color="#F0F0F0", 
                        markersize=4.0, alpha=0.9, zorder=dot_zorder + 1, label=lbl)

        # --- 4. AESTHETICS ---
        ax.dist = 11
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) 
        
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
        
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.view_init(elev=28, azim=-115) 
        files = glob.glob("results/calibration_*_prices.csv")
        files_sorted = sorted(files, key=os.path.getctime)
        if files:
            # Pick the latest file
            latest_file = files_sorted[2] #max(files, key=os.path.getctime)
        csv_file = pd.read_csv(latest_file)
        base_name = csv_file.replace("_prices.csv", "")
        json_file = f"{base_name}_meta.json"
        try:
            if "calibration_" in base_name:
                ticker = base_name.split("calibration_")[1].split("_")[0]

            with open(json_file, 'r') as f:
                meta = json.load(f)
                s0 = meta['market']['S0']
                params = meta.get('analytical', {})
                # Also check if params are in a deeper 'bates' key or similar, 
                # but usually they are at the top of 'analytical'
        except Exception:
            pass    
        s0 = 278.12 #6923.54 
        # --- TITLES ---
        model_name = "Bates" if is_bates else "Heston"
        fig.text(0.535, 0.84, rf"{model_name} Implied Volatility Surface: AAPL", 
                 color='white', fontsize=16, fontweight='bold', family='monospace', ha='center')
        
        if is_bates:
            subtitle = (rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$" + "\n" +
                        rf"$\lambda={lamb:.2f}, \mu_J={mu_j:.2f}, \sigma_J={sigma_j:.2f}, S_0={s0:.1f}$")
            fig.text(0.535, 0.79, subtitle, color='#AAAAAA', fontsize=9, family='monospace', ha='center')
        else:
            subtitle = rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$"
            fig.text(0.535, 0.81, subtitle, color='#AAAAAA', fontsize=10, family='monospace', ha='center')

        # --- 5. PERFORMANCE METRICS ---
        obj_val = params.get('weighted_obj', params.get('fun', 0))
        rmse_val = params.get('rmse', params.get('rmse_iv', 0))

        comparison_text = (
            f"Model Source: {source_name}\n"
            f"-------------------\n"
            f"Final RMSE ($):     {rmse_val:.4f}\n"
            f"Obj Function:       {obj_val:.4f}\n"
            f"Outliers Removed:   {dropped_count}"
        )
        print("\n" + comparison_text)
        
        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=10)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=10)
        ax.set_zlabel(r'Implied Volatility', color='white', labelpad=10)

        if market_options and valid_needles > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.157, 0.797), frameon=False, labelcolor="#D7D7D7", fontsize=10)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.outline.set_visible(False)

        save_path = f"{filename}_surface_refined.png"
        plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
        print(f"-> Saved: {save_path}")
        #plt.show()

def main():
    try:
        data, r_curve, q_curve, market_options, base_name = load_latest_calibration()
        S0 = data['market']['S0']
        
        best_params, source_name = select_best_parameters(data)
        ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
        
        print(f"\n[Direct Plot] Using {source_name} parameters directly.")
        
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