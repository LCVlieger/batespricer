import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedLocator
import io
from PIL import Image

# Local package imports
try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass



def get_premium_style():
    """
    Combines Custom 'Nordic' Hex codes with 3-Segment Non-Linear Warping.
    """
    # 1. THE PALETTE (Clean, report-ready colors)
    # We define a custom linear map first, covering 0.0 to 1.0
    colors = [
        "#5b85d9",  # 0.0: Strong Cornflower
        "#b4d3f7",  # 0.25: Glacial Ice
        "#ffffff",  # 0.5: PURE WHITE
        "#fcd34d",  # 0.75: Soft Gold
        "#d97706"   # 1.0: Deep Amber
    ]
    base_cmap = mcolors.LinearSegmentedColormap.from_list('nordic_base', colors, N=256)
    
    # 2. THE WARPING LOGIC (Your Code, Tuned)
    N = 256
    # We use 0.0 to 1.0 because our custom map is already perfect. 
    # No need to slice it like 'jet'.
    values = np.linspace(0.0, 1.0, N) 
    
    # --- Thresholds (Tuned for this palette) ---
    # t1: Where the "Blue" stops and "White" begins
    # t2: Where the "White" stops and "Gold" begins
    t1 = 0.35 
    t2 = 0.55 
    
    # --- Gammas ---
    g_floor = 1.1   # Slight compression of the blue
    # We don't need complex gamma for the middle because our map has White at 0.5
    # But we keep your logic for the curvature.
    
    warped_values = np.zeros_like(values)

    # Segment 1: The Floor (Blue -> Ice)
    mask1 = values <= t1
    if np.any(mask1):
        s1 = values[mask1] / t1
        warped_values[mask1] = (s1 ** g_floor) * t1

    # Segment 2: The Slope (Ice -> White -> Gold)
    # This is the most critical part for the "Light Background" look.
    mask2 = (values > t1) & (values <= t2)
    if np.any(mask2):
        s2 = (values[mask2] - t1) / (t2 - t1)
        # Your Smoothstep logic (Quintic)
        s2_smooth = s2 * s2 * s2 * (s2 * (s2 * 6.0 - 15.0) + 10.0)
        # We relax the gamma here so the white doesn't disappear too fast
        s2_weighted = np.power(s2_smooth, 0.8) 
        warped_values[mask2] = s2_weighted * (t2 - t1) + t1

    # Segment 3: The Peaks (Gold -> Amber)
    mask3 = values > t2
    if np.any(mask3):
        v_max = 1.0
        s3 = np.maximum(0, (values[mask3] - t2) / (v_max - t2))
        # Sine Ease-Out
        s3_sine = np.sin(s3 * np.pi / 2.0)
        warped_values[mask3] = s3_sine * (v_max - t2) + t2

    # 3. APPLY WARPING TO COLORS
    warped_values = np.clip(warped_values, 0, 1)
    new_colors = base_cmap(warped_values)
    
    return mcolors.ListedColormap(new_colors, name='Nordic_Warped')


def create_premium_cmap_1():
    """
    'Neon Surface' Colormap.
    Specifically calibrated for Black Backgrounds.
    Lifts the 'floor' luminance so the blue does not disappear.
    """
    #colors = [
    #    "#0F3CB7",  # Bright Dodger Blue
    #      # Deep transition
    #    "#3498DB",  # Mid Blue
    #    "#00C6FF",  # Laser Cyan
    #    "#E0F7FA",  # Icy White (The 'Shine' point)
    #    "#FFF176",  # Champagne 
    #    "#FFC107"   # Amber/Gold (The Peak)
    #]
    colors = [
            "#081B4B",  # Deep Midnight/Navy (The deep OTM/ITM base)
            "#0F3CB7",  # Bright Dodger Blue
            "#3498DB",  # Mid Blue
            "#00C6FF",  # Laser Cyan (Transition)
            "#B2EBF2",  # Pale Cyan 
            "#E0F7FA"   # Icy White (The Peak / highest volatility)
        ]
    # We position the nodes to give the blue floor more space, 
    # ensuring the whole surface looks illuminated.
    nodes = [0.0, 0.1, 0.3, 0.45, 0.8, 1.0]
    
    cmap = mcolors.LinearSegmentedColormap.from_list("NeonGold", list(zip(nodes, colors)))
    base = cm.get_cmap(cmap)
    gamma=0.8
    N = 256
    values = np.linspace(0, 0.75, N)
    # Apply gamma correction to the indices we pull from the original map
    # gamma < 1 stretches the low end (makes it pop)
    warped_values = values ** gamma 
    return cmap

stale_red ="#9E0000"

def create_mint_surface_cmap():
    """
    Creates a 'Minty' colormap (Deep Teal -> Green -> Yellow).
    Designed specifically to make Magenta/Red dots pop without edges.
    """
    # Hex codes for the Surface (The "Cool" Side)
    # We avoid red/purple here completely.
    colors = [
        "#0f4c5c",  # Deep Petrol/Teal (Floor) - Dark enough to anchor, but green-tinted
        "#267d73",  # Jungle Green
        "#5fba7d",  # Medium Sea Green
        "#9ce072",  # Vivid Lime
        "#f2e94e"   # Lemon Yellow (Peak)
    ]
    
    # 1. Create Base Map
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('mint_quant', colors, N=n_bins)
    
    # 2. Apply "Warping" (Gamma Correction)
    # We use your logic to stretch the 'floor' so the deep teal covers the low-vol areas
    # This ensures the floor isn't too bright, providing contrast for the background.
    vals = np.linspace(0, 1, n_bins)
    gamma = 0.9  # Slight adjustment to keep the mid-tones vibrant
    warped_vals = np.power(vals, gamma)
    
    warped_colors = cmap(warped_vals)
    return mcolors.ListedColormap(warped_colors, name='mint_quant_warped')

def create_paper_premium_cmap(style='plasma_fire'):
    """
    Creates a custom colormap designed specifically for white backgrounds.
    It avoids 'invisible' yellows and maximizes saturation.
    """
    
    # 1. 'Plasma Fire': Deep Purple -> Red -> Orange (Classic Vol Surface)
    # Good for: Highlighting extreme volatility without losing the floor.
    if style == 'plasma_fire':
        colors = [
            "#0d0887",  # Deep Indigo (Floor - distinct from white)
            "#7201a8",  # Rich Purple
            "#bd3786",  # Magenta
            "#ed7953",  # Salmon
            "#fdc527",  # Deep Gold (Not pale yellow!)
            "#f0f921"   # Brightest Yellow (Only for very peaks)
        ]
        
    # 2. 'Oceanic Tech': Deep Navy -> Teal -> Lime (Modern/Clean)
    # Good for: Smooth surfaces, very 'fintech' look.
    elif style == 'oceanic':
        colors = [
            "#081d58",  # Midnight Blue (Floor)
            "#225ea8",  # Royal Blue
            "#41b6c4",  # Ocean Teal
            "#7fcdbb",  # Soft Cyan
            "#c7e9b4",  # Pale Green (Mid)
            "#edf8b1"   # Pale Yellow (Peak)
        ]

    # 3. 'Cyber-Paper': The "Cool" option. 
    # Deep Blue -> Cyan -> White/Grey -> Red -> Dark Red
    # Mimics a diverging map but keeps the 'cool' intensity.
    elif style == 'cyber_paper':
        colors = [
            "#202060",  # Deep Slate Blue (Floor)
            "#4060b0",  # Medium Blue
            "#60a0d0",  # Steel Blue
            "#b0d0e0",  # Ice Blue
            "#f0b090",  # Soft Orange
            "#d05040",  # Brick Red
            "#801010"   # Deep Crimson (Peak)
        ]

    # Create the base linear map
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list(f'custom_{style}', colors, N=n_bins)
    
    # --- APPLY YOUR WARPING LOGIC ---
    # We apply a gamma correction to 'push' the middle colors 
    # so the floor doesn't dominate.
    
    # Extract the colors from the new map
    vals = np.linspace(0, 1, n_bins)
    
    # Gamma < 1 expands the darks (makes more floor visible)
    # Gamma > 1 expands the lights (makes more peaks visible)
    # For white paper, we usually want Gamma ~0.8 to make sure the faint blues show up.
    gamma = 0.85 
    warped_vals = np.power(vals, gamma)
    
    # Re-sample the colormap
    warped_colors = cmap(warped_vals)
    
    return mcolors.ListedColormap(warped_colors, name=f'{style}_warped')
def create_premium_cmap(base_cmap_name):
    base = cm.get_cmap(base_cmap_name)
    N = 256
    values = np.linspace(0.007125, 0.6485, N) #0.02 195
    
    # --- Define Thresholds --- 
    t1 = 0.1  # End of the 'Floor' (Blue) 0.12
    t2 = 0.21  # Start of the 'Peaks' (Gold)
    
    # --- Define Gammas ---
    g_floor = 1.3  # >1 compresses the dark blue (keeps it at the bottom)
    g_slope = 0.78  # <1 expands the 'glow' transition (Cyan/White)
    g_peak  = 0.7  # <1 expands the gold at the very top for lighting
    
    warped_values = np.zeros_like(values)

    # 1. Segment One: The Floor (0 to t1)
    mask1 = values <= t1
    s1 = values[mask1] / t1
    warped_values[mask1] = (s1 ** g_floor) * t1

    # 2. Segment Two: The Slope (t1 to t2)
    mask2 = (values > t1) & (values <= t2)
    if np.any(mask2):
        s2 = (values[mask2] - t1) / (t2 - t1)
        s2_smooth = s2 * s2 * s2 * (s2 * (s2 * 6.0 - 15.0) + 10.0)
        gamma = 0.3
        s2_weighted = np.power(s2_smooth, gamma)
        warped_values[mask2] = s2_weighted * (t2 - t1) + t1

    # 3. Segment Three: The Peaks (t2 to max) - Sine Ease-Out for vibrant highlights
    mask3 = values > t2
    if np.any(mask3):
        v_max = values.max()
        s3 = np.maximum(0, (values[mask3] - t2) / (v_max - t2))
        s3_sine = np.sin(s3 * np.pi / 2.0)
        warped_values[mask3] = s3_sine * (v_max - t2) + t2

    warped_values = np.clip(warped_values, 0, 1)
    colors = base(warped_values)
    
    return mcolors.ListedColormap(colors, name=f'Warped_3Seg_{base_cmap_name}')

class RobustYieldCurve:
    def __init__(self, curve_data):
        times, rates = [], []
        if isinstance(curve_data, dict):
            for k, v in curve_data.items():
                try:
                    t_str = str(k).lower().replace("y", "").replace("week", "")
                    times.append(float(t_str))
                    rates.append(float(v))
                except: 
                    continue
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

def load_calibration_by_index(index):
    patterns = ["results/*_meta.json", "*_meta.json"]
    files = []
    for p in patterns: 
        files.extend(glob.glob(p))
        
    if not files: 
        raise FileNotFoundError("No calibration meta file found.")
        
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    
    if index >= len(files_sorted):
        return None 
    latest_meta = files_sorted[index]
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading Artifact: {base_name}...")

    with open(latest_meta, 'r') as f: 
        data = json.load(f)

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
    return res_mc, "Analytical"

def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    lamb = params.get('lamb', 0.0)
    mu_j = params.get('mu_j', 0.0)
    sigma_j = params.get('sigma_j', 0.0)
    is_bates = lamb > 0.0

    LOWER_M, UPPER_M = 0.685, 1.315                    
    LOWER_T, UPPER_T = 0.04, 1.5 
    GRID_DENSITY =  60 # 550# 550 #80

    print(f"-> Generating Surface for: {ticker}")
    print(f"   Model: {'Bates' if is_bates else 'Heston'}")
    print(f"   Calculating true gradient-based adaptive mesh...")
    
    COARSE_N = 30 # 120 #80  150
    c_M = np.linspace(LOWER_M, UPPER_M, COARSE_N)
    c_T = np.linspace(LOWER_T, UPPER_T, COARSE_N)
    cX, cY = np.meshgrid(c_M, c_T)
    cZ = np.zeros_like(cX)

    for i in range(COARSE_N):
        for j in range(COARSE_N):
            T_val, M_val = cY[i, j], cX[i, j]
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)
            try:
                prices = BatesAnalyticalPricer.price_european_call_vectorized(
                    S0, np.array([S0 * M_val]), np.array([T_val]), np.array([r_T]), np.array([q_T]),
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                )
                iv = implied_volatility(float(prices[0]), S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                cZ[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                cZ[i, j] = np.nan

    mask = np.isnan(cZ)
    if np.any(mask):
        cZ = pd.DataFrame(cZ).interpolate(method='linear', axis=1, limit_direction='both') \
                             .interpolate(method='linear', axis=0, limit_direction='both').values

    DENSITY_POWER = 2.3
    if cZ is not None and hasattr(cZ, 'shape') and cZ.shape[0] >= 2 and cZ.shape[1] >= 2:
        try:
            dZ_dT, dZ_dM = np.gradient(cZ, c_T, c_M)
            grad_mag = np.sqrt(dZ_dT**2 + dZ_dM**2)
            max_grad = np.percentile(grad_mag, 92) 
            grad_mag = np.clip(grad_mag, 0, max_grad)
            dens_M = np.mean(grad_mag, axis=0)**DENSITY_POWER
            dens_T = np.mean(grad_mag, axis=1)**DENSITY_POWER
        except ValueError:
            dens_M, dens_T = np.ones(len(c_M)), np.ones(len(c_T))
    else:
        dens_M, dens_T = np.ones(len(c_M)), np.ones(len(c_T))

    def get_hybrid_spacing(density_array, grid_points, mix_ratio=0.7):
        cdf_grad = np.cumsum(density_array)
        if cdf_grad[-1] - cdf_grad[0] == 0:
            cdf_grad = np.linspace(0, 1, len(density_array))
        else:
            cdf_grad = (cdf_grad - cdf_grad[0]) / (cdf_grad[-1] - cdf_grad[0])
            
        cdf_linear = np.linspace(0, 1, len(density_array))
        return (mix_ratio * cdf_grad) + ((1 - mix_ratio) * cdf_linear)

    cdf_M_final = get_hybrid_spacing(dens_M, COARSE_N, mix_ratio=0.7)
    cdf_T_final = get_hybrid_spacing(dens_T, COARSE_N, mix_ratio=0.7)
    
    uniform_space = np.linspace(0, 1, GRID_DENSITY)
    M_range = np.interp(uniform_space, cdf_M_final, c_M)
    T_range = np.interp(uniform_space, cdf_T_final, c_T)

    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)
            try:
                prices = BatesAnalyticalPricer.price_european_call_vectorized(
                    S0, np.array([S0 * M_val]), np.array([T_val]), np.array([r_T]), np.array([q_T]),
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                )
                iv = implied_volatility(float(prices[0]), S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                Z[i, j] = np.nan

    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1, limit_direction='both') \
                           .interpolate(method='linear', axis=0, limit_direction='both').values
        
    Z_smooth = gaussian_filter(Z, sigma=0.5)
    
    def create_gamma_cmap(base_cmap_name, gamma=0.5):
        base = cm.get_cmap(base_cmap_name)
        N = 256
        values = np.linspace(0, 1.0, N)
        warped_values = values ** gamma 
        colors = base(warped_values)
        return mcolors.ListedColormap(colors, name=f'Warped_{base_cmap_name}')

    # --- CHANGED: Removed 'dark_background' context, set facecolor to white ---
    fig = plt.figure(figsize=(10, 7), facecolor='white') 
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    ls = LightSource(azdeg=270, altdeg=45)
    vmin, vmax = 0.1151, 0.72
    my_cmap = create_gamma_cmap('viridis', gamma=0.5) 
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    rgb = ls.shade(Z_smooth, cmap=my_cmap, norm=norm, vert_exag=0.1)
    
    surf = ax.plot_surface(X, Y, Z_smooth, facecolors=rgb, cmap=my_cmap, 
                           rcount=X.shape[0], ccount=X.shape[1], 
                           edgecolor='none', linewidth=0.2, alpha=0.6, 
                           shade=False, antialiased=True, zorder=1, rasterized=True)
                               
    m = cm.ScalarMappable(cmap=my_cmap, norm=norm)
    m.set_array([])
    
    # --- CHANGED: Stale Red Dot Styling ---
    # "#D90429"

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
            alpha_above = 0.85  # Keep these crisp
            alpha_below = 1.0  # Make these very transparent

            current_alpha = alpha_above if is_above else alpha_below
            
            # CHANGED: Needles are now stale_red instead of white
            ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod_exact, iv_mkt], 
                    color=stale_red, linestyle='-', linewidth=0.8, alpha=0.65, zorder=dot_zorder)
            lbl = 'Market IV' if valid_needles == 1 else ""
            
            # CHANGED: Dots are now stale_red instead of F0F0F0
            ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mkt], 
                    marker='o', linestyle='None', color=stale_red, markersize=4.62,
                    markerfacecolor=stale_red, markeredgecolor='none', markeredgewidth=0.01,
                    alpha=current_alpha, zorder=dot_zorder + 1, label=lbl)
                        
    ax.dist = 11  
    ax.set_xlim(LOWER_M, UPPER_M)
    ax.set_ylim(UPPER_T, LOWER_T) 
    ax.set_zlim(0.0, 0.75)
    
    # --- CHANGED: Light Theme Grid and Pane Styling ---
    grid_style = (0.68, 0.68, 0.68, 0.5) 
    linewidth_val = 0.575
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((0.95, 0.95, 0.95, 1.0)) # Light grey pane
        axis.line.set_color("black")  # Changed axis line to black
        axis.line.set_linewidth(0.8)
        axis._axinfo["grid"]['color'] = grid_style 
        axis._axinfo["grid"]['linewidth'] = linewidth_val

    ax.view_init(elev=28, azim=-115) 

    # --- CHANGED: Label colors to black ---
    ax.set_xlabel('Moneyness ($K/S_0$)', color="black", labelpad=5, fontsize=11)
    ax.set_ylabel('Maturity ($T$ Years)', color="black", labelpad=5, fontsize=11)
    ax.set_zlabel(r'Implied Volatility', color="black", labelpad=6.75, fontsize=11)
    ax.tick_params(axis='both', which='major', colors='black', labelsize=10)

    if market_options and valid_needles > 0:
        # CHANGED: Legend styling to match light theme
        ax.legend(loc='upper left', bbox_to_anchor=(0.175, 0.79), frameon=True, 
                  facecolor=(0.95, 0.95, 0.95), labelcolor="black", handletextpad=0.5, edgecolor='none', fontsize=10)
        leg = ax.get_legend()
        for handle in leg.legend_handles:
            handle.set_alpha(1)

    # --- THE REAL SOLUTION: PURE VIBRANT COLORBAR ---
    cb_values = np.linspace(vmax, vmin, 256)
    cb_base_colors = my_cmap(norm(cb_values)) 

    cb_hsv = mcolors.rgb_to_hsv(cb_base_colors[:, :3])
    cb_hsv[:, 1] = np.clip(cb_hsv[:, 1] * 1.2, 0, 1) 
    cb_hsv[:, 2] = np.clip(cb_hsv[:, 2] * 1.05, 0, 1) 
    
    cb_rgb_vibrant = mcolors.hsv_to_rgb(cb_hsv)

    cb_rgba_final = np.zeros((256, 1, 4))
    cb_rgba_final[:, 0, :3] = cb_rgb_vibrant
    cb_rgba_final[:, 0, 3] = 0.85 

    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=15, pad=-0.02)
    cbar.ax.clear()
    
    cbar.ax.imshow(cb_rgba_final, aspect='auto', extent=[0, 1, vmin, vmax], 
                   origin='upper', interpolation='bilinear')
    
    cbar.ax.set_rasterized(True) 
    cbar.ax.xaxis.set_visible(False)
    cbar.ax.set_frame_on(False)

    cbar.locator = FixedLocator(np.arange(0.1, 0.8, 0.1))
    cbar.update_ticks()
    # --- CHANGED: Colorbar ticks and title to black ---
    cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black", labelsize=10, width=0.5)
    cbar.outline.set_visible(False)
    cbar.ax.set_title("Model IV", color="black", fontsize=10, pad=9)
    
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

    save_path_vector = f"{filename}_surface_FINAL_LIGHT.pdf"
    # --- CHANGED: facecolor='white' for saving ---
    plt.savefig(save_path_vector, format='pdf', bbox_inches='tight',    
                pad_inches=0.15, facecolor='white', dpi=800)

def main():
    num_to_plot = 2 
    for i in range(num_to_plot):
        try:
            print(f"\n--- Processing Artifact {i+1} ---")
            result = load_calibration_by_index(i) 
            if result is None:
                print(f"No file found for index {i}. Skipping.")
                continue
                
            data, r_curve, q_curve, market_options, base_name = result
            S0 = data['market']['S0']
            best_params, source_name = select_best_parameters(data)
            
            ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
            
            plot_surface_professional(
                S0, r_curve, q_curve, best_params, ticker, base_name, 
                market_options, data, 0, source_name
            )
        except Exception as e:
            print(f"[Error at index {i}] {e}")

if __name__ == "__main__":
    main()