import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from scipy.ndimage import gaussian_filter 
from scipy.interpolate import interp1d

# Local package imports
try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource

def create_gamma_cmap(base_cmap_name, gamma=0.5):
    base = cm.get_cmap(base_cmap_name)
    N = 256
    # Create a linear ramp
    values = np.linspace(0, 1, N)
    # Apply gamma correction to the indices we pull from the original map
    # gamma < 1 stretches the low end (makes it pop)
    warped_values = values ** gamma 
    colors = base(warped_values)
    return mcolors.ListedColormap(colors, name=f'Warped_{base_cmap_name}')

# --- 1. COMPATIBILITY HELPERS ---
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
            times = curve_data.tenors
            rates = curve_data.rates
        else:
            times = [0.0, 30.0]
            rates = [float(curve_data), float(curve_data)]

        sorted_pairs = sorted(zip(times, rates))
        self.ts = np.array([p[0] for p in sorted_pairs])
        self.rs = np.array([p[1] for p in sorted_pairs])
        self.interp = interp1d(self.ts, self.rs, kind='linear', 
                               bounds_error=False, fill_value=(self.rs[0], self.rs[-1]))

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
    latest_meta = files_sorted[-1]
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading Artifact: {base_name}...")
    
    with open(latest_meta, 'r') as f: data = json.load(f)
    
    r_data = data['market'].get('r_sample', data['market'].get('r'))
    q_data = data['market'].get('q_sample', data['market'].get('q'))
    
    r_curve = RobustYieldCurve(r_data)
    q_curve = RobustYieldCurve(q_data)

    csv_file = f"{base_name}_prices.csv"
    market_options = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            k = row.get('K', row.get('Strike', 0))
            t = row.get('T', row.get('Maturity', 0))
            p = row.get('Market', row.get('Price', 0))
            otype = row.get('Type', "CALL")
            market_options.append(ReconstructedOption(k, t, p, otype))

    return data, r_curve, q_curve, market_options, base_name

def select_best_parameters(data):
    res_mc = data.get('analytical', {})
    return res_mc, "Monte Carlo"

# --- 2. EXACT PLOTTING FUNCTION ---
def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    lamb = params.get('lamb', 0.0)
    mu_j = params.get('mu_j', 0.0)
    sigma_j = params.get('sigma_j', 0.0)
    is_bates = lamb > 0.0

    # --- CONFIGURATION ---
    LOWER_M, UPPER_M = 0.685, 1.315 
    LOWER_T, UPPER_T = 0.04, 1.5 
    GRID_DENSITY = 47

    M_range = np.linspace(LOWER_M, UPPER_M, GRID_DENSITY)
    T_range = np.linspace(LOWER_T, UPPER_T, GRID_DENSITY)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    print(f"-> Generating Surface for: {ticker}")
    print(f"   Model: {'Bates' if is_bates else 'Heston'}")
    
    # --- CALCULATION ---
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)
            
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

    # --- PLOTTING ---
    with plt.style.context('dark_background'):
        # Keep the chunkier size (10, 7) for better font scaling
        fig = plt.figure(figsize=(10, 7), facecolor='black') 
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        from matplotlib.colors import LightSource
        import matplotlib.colors as mcolors
        ls = LightSource(azdeg=270, altdeg=45)
        vmin = 0.0953 
        vmax=0.72
        my_cmap = create_gamma_cmap('RdYlBu_r', gamma=1.3)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # Use the light source to shade the data
        rgb = ls.shade(Z, cmap=my_cmap, norm=norm, vert_exag=0.1)
        surf = ax.plot_surface(X, Y, Z_smooth, facecolors = rgb,    cmap=cm.RdYlBu_r, 
                               rcount=100, ccount=100,  
                               edgecolor='black', linewidth=0.0, alpha=0.8,#0.085   #0.8                     
                               shade=False, antialiased=True, zorder=1)
        norm_bar = mcolors.Normalize(vmin=vmin, vmax=vmax)
        m = cm.ScalarMappable(cmap=my_cmap, norm=norm)
        m.set_array([])
        if market_options:
            plot_opts = [o for o in market_options 
                         if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)]
            
            valid_needles = 0
            for opt in plot_opts:
                m_mkt, t_mkt = opt.strike / S0, opt.maturity
                try:
                    r_T_mkt = r_curve.get_rate(t_mkt)
                    q_T_mkt = q_curve.get_rate(t_mkt)
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)
                    prices_mod = BatesAnalyticalPricer.price_vectorized(
                        S0, np.array([opt.strike]), np.array([t_mkt]), np.array([r_T_mkt]), np.array([q_T_mkt]), np.array([opt.option_type]),
                        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                    )
                    iv_mod_exact = implied_volatility(float(prices_mod[0]), S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)
                    if iv_mkt < 0.01 or iv_mkt > 2.5: continue
                except: continue

                valid_needles += 1
                is_above = iv_mkt >= iv_mod_exact
                dot_zorder = 10 if is_above else 1

                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod_exact, iv_mkt], 
                        color='white', linestyle='-', linewidth=0.8, alpha=0.65, zorder=dot_zorder)
                lbl = 'Market Price-IV' if valid_needles == 1 else ""
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color="#F0F0F0", 
                        markersize=4.0, alpha=0.9, zorder=dot_zorder + 1, label=lbl)

        # --- AESTHETICS ---
        ax.dist = 11  # Your preferred zoom level
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) 
        ax.set_zlim(0.0, 0.75)
        grid_style = (0.23, 0.23, 0.23, 0.82) #(0.55, 0.55, 0.55, 0.35) 
        linewidth_val = 1.7925
        ax.xaxis.set_pane_color((0, 0, 0, 1))
        ax.yaxis.set_pane_color((0, 0, 0, 1))
        ax.zaxis.set_pane_color((0, 0, 0, 1))
        ax.xaxis._axinfo["grid"]['color'] = grid_style 
        ax.yaxis._axinfo["grid"]['color'] = grid_style 
        ax.zaxis._axinfo["grid"]['color'] = grid_style 
        ax.xaxis._axinfo["grid"]['linewidth'] = linewidth_val
        ax.yaxis._axinfo["grid"]['linewidth'] = linewidth_val
        ax.zaxis._axinfo["grid"]['linewidth'] = linewidth_val
        ax.view_init(elev=28, azim=-115) 

        # Axis Labels: Padded to prevent overlap with tick labels
        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=5, fontsize=11)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=5, fontsize=11)
        ax.set_zlabel(r'Implied Volatility', color='white', labelpad=5, fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # --- TITLES ---
        model_name = "Bates" if is_bates else "Heston"
        
        # FIX 1: Lower y-position to 0.78 to meet the zoomed-out (dist=11) plot
        # FIX 2: Shift x to 0.52 to correct for 3D perspective shift 0.537675
        #fig.text(0.55, 0.811, rf"{model_name} Implied Volatility Surface: ^SPX", 
        #         color='white', fontsize=16, fontweight='bold', family='monospace', ha='center')

        # --- LEGEND ---
        if market_options and valid_needles > 0:
            # FIX 3: Safe coordinates that won't float away
            ax.legend(loc='upper left', bbox_to_anchor=(0.175, 0.79), 
                      frameon=False, labelcolor="#D7D7D7", fontsize=10)
        from matplotlib.ticker import FixedLocator
        # --- COLORBAR ---
        cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=15, pad=-0.02)
        tick_locations = np.arange(0.1, 0.8, 0.1) # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        cbar.locator = FixedLocator(tick_locations)
        cbar.update_ticks()
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=10)
        cbar.outline.set_visible(False)
        cbar.ax.set_title("Model IV", color='white', fontsize=10, pad=9)
        
        save_path = f"{filename}_surface_refined.png"
        
        # FIX 4: Add pad_inches=0.2 to prevent slicing off labels
        # 1. Save to a temporary buffer first
        import io
        from PIL import Image

        # --- AESTHETICS ---
        ax.dist = 11 
        ax.view_init(elev=28, azim=-115) 
        
        # 1. THE "NATIVE CROP": Adjust the subplot to fill the canvas
        # This removes the "blank space" at the top and right manually
        fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

        # 2. SAVE DIRECTLY TO VECTOR PDF
        # We skip PIL entirely to preserve mathematical sharpness
        save_path_vector = f"{filename}_surface_FINAL.pdf"
        
        plt.savefig(save_path_vector, 
                    format='pdf', 
                    bbox_inches='tight', 
                    pad_inches=0.05, 
                    facecolor='black')
        
        print(f"-> Saved True Vector: {save_path_vector}")

def main():
    try:
        data, r_curve, q_curve, market_options, base_name = load_latest_calibration()
        S0 = data['market']['S0']
        
        best_params, source_name = select_best_parameters(data)
        ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
        
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