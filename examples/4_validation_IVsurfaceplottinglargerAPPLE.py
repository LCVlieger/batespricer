import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedLocator

# Local package imports
try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass

def create_gamma_cmap(base_cmap_name, gamma=0.5):
    base = cm.get_cmap(base_cmap_name)
    return mcolors.LinearSegmentedColormap.from_list(
        f'Warped_{base_cmap_name}',
        [base(x**gamma) for x in np.linspace(0, 1, 1024)]
    )

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
        else:
            times, rates = [0.0, 30.0], [float(curve_data), float(curve_data)]
            
        sorted_pairs = sorted(zip(times, rates))
        self.ts, self.rs = np.array([p[0] for p in sorted_pairs]), np.array([p[1] for p in sorted_pairs])
        self.interp = interp1d(self.ts, self.rs, kind='linear', bounds_error=False, fill_value=(self.rs[0], self.rs[-1]))

    def get_rate(self, T):
        return float(self.interp(max(T, 1e-4)))

class ReconstructedOption:
    def __init__(self, strike, maturity, price, option_type="CALL"):
        self.strike, self.maturity, self.market_price, self.option_type = float(strike), float(maturity), float(price), str(option_type)

def load_latest_calibration():
    files = glob.glob("results/*_meta.json") + glob.glob("*_meta.json")
    if not files: raise FileNotFoundError("No calibration meta file found.")
    latest_meta = sorted(files, key=os.path.getctime)[-1]
    base_name = latest_meta.replace("_meta.json", "")
    with open(latest_meta, 'r') as f: data = json.load(f)
    r_curve, q_curve = RobustYieldCurve(data['market'].get('r_sample', data['market'].get('r'))), RobustYieldCurve(data['market'].get('q_sample', data['market'].get('q')))
    
    market_options = []
    csv_file = f"{base_name}_prices.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            market_options.append(ReconstructedOption(row.get('K', row.get('Strike', 0)), row.get('T', row.get('Maturity', 0)), row.get('Market', row.get('Price', 0)), row.get('Type', "CALL")))
    return data, r_curve, q_curve, market_options, base_name

def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full):
    # Model Params
    p = params
    k, th, xi, rho, v0 = p['kappa'], p['theta'], p['xi'], p['rho'], p['v0']
    l, mj, sj = p.get('lamb', 0.0), p.get('mu_j', 0.0), p.get('sigma_j', 0.0)

    # Config
    LOWER_M, UPPER_M = 0.685, 1.315 
    LOWER_T, UPPER_T = 0.04, 1.5 
    GRID_DENSITY = 60 # Final resolution

    # --- PHASE 1: GEOMETRIC ADAPTIVE MESH ---
    # Coarse pass to find curvature
    COARSE_N = 30
    c_m, c_t = np.linspace(LOWER_M, UPPER_M, COARSE_N), np.linspace(LOWER_T, UPPER_T, COARSE_N)
    cX, cY = np.meshgrid(c_m, c_t)
    cZ = np.zeros_like(cX)

    for i in range(COARSE_N):
        for j in range(COARSE_N):
            T, M = cY[i,j], cX[i,j]
            price = BatesAnalyticalPricer.price_european_call_vectorized(S0, np.array([S0*M]), np.array([T]), np.array([r_curve.get_rate(T)]), np.array([q_curve.get_rate(T)]), k, th, xi, rho, v0, l, mj, sj)[0]
            try: iv = implied_volatility(float(price), S0, S0*M, T, r_curve.get_rate(T), q_curve.get_rate(T), "CALL")
            except: iv = np.nan
            cZ[i,j] = iv if 0.01 < iv < 2.5 else np.nan
    
    cZ = pd.DataFrame(cZ).interpolate(axis=1).ffill(axis=1).bfill(axis=1).values
    
    # Calculate Curvature-based Monitor Function
    dZ_dT, dZ_dM = np.gradient(cZ)
    d2Z_dM2 = np.abs(np.gradient(dZ_dM, axis=1))
    d2Z_dT2 = np.abs(np.gradient(dZ_dT, axis=0))
    
    # Weight nodes by second derivative (curvature)
    w_m = np.sqrt(1 + 10.0 * d2Z_dM2.max(axis=0))
    w_t = np.sqrt(1 + 10.0 * d2Z_dT2.max(axis=1))

    def equidistribute(nodes, weights, n):
        cdf = np.cumsum(weights)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        return interp1d(cdf, nodes, kind='linear', fill_value="extrapolate")(np.linspace(0, 1, n))

    M_range = equidistribute(c_m, w_m, GRID_DENSITY)
    T_range = equidistribute(c_t, w_t, GRID_DENSITY)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- PHASE 2: FINAL COMPUTE ---
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T, M = Y[i,j], X[i,j]
            rt, qt = r_curve.get_rate(T), q_curve.get_rate(T)
            price = BatesAnalyticalPricer.price_european_call_vectorized(S0, np.array([S0*M]), np.array([T]), np.array([rt]), np.array([qt]), k, th, xi, rho, v0, l, mj, sj)[0]
            try: Z[i,j] = implied_volatility(float(price), S0, S0*M, T, rt, qt, "CALL")
            except: Z[i,j] = np.nan

    Z = pd.DataFrame(Z).interpolate(axis=1).ffill(axis=1).bfill(axis=1).values
    
    # --- PHASE 3: RENDERING ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(10, 7), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        # High-res shading
        zoom_f = 2
        Z_h, X_h, Y_h = zoom(Z, zoom_f, order=3), zoom(X, zoom_f, order=3), zoom(Y, zoom_f, order=3)
        ls = LightSource(azdeg=270, altdeg=45)
        my_cmap = create_gamma_cmap('RdYlBu_r', gamma=1.1)
        norm = mcolors.Normalize(vmin=0.115, vmax=0.72)
        rgb = ls.shade(Z_h, cmap=my_cmap, norm=norm, vert_exag=0.1)
        
        ax.plot_surface(X_h, Y_h, Z_h, facecolors=rgb, shade=False, antialiased=True, alpha=0.8, rasterized=True)

        # Market Points
        if market_options:
            for opt in market_options:
                m_mkt, t_mkt = opt.strike/S0, opt.maturity
                if LOWER_M <= m_mkt <= UPPER_M and LOWER_T <= t_mkt <= UPPER_T:
                    rt, qt = r_curve.get_rate(t_mkt), q_curve.get_rate(t_mkt)
                    try:
                        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, t_mkt, rt, qt, opt.option_type)
                        p_mod = BatesAnalyticalPricer.price_vectorized(S0, np.array([opt.strike]), np.array([t_mkt]), np.array([rt]), np.array([qt]), np.array([opt.option_type]), k, th, xi, rho, v0, l, mj, sj)[0]
                        iv_mod = implied_volatility(float(p_mod), S0, opt.strike, t_mkt, rt, qt, opt.option_type)
                        ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod, iv_mkt], color='white', linewidth=0.8, alpha=0.5)
                        ax.scatter(m_mkt, t_mkt, iv_mkt, color='#F0F0F0', s=20, edgecolors='none', alpha=0.8)
                    except: continue

        # Styling
        ax.view_init(elev=28, azim=-115)
        ax.set_xlabel('Moneyness ($K/S_0$)'); ax.set_ylabel('Maturity ($T$)'); ax.set_zlabel('IV')
        ax.set_zlim(0, 0.75); ax.dist = 11
        
        plt.savefig(f"{filename}_surface_FINAL.pdf", format='pdf', bbox_inches='tight', facecolor='black', dpi=600)
        print(f"-> Exported: {filename}_surface_FINAL.pdf")

def main():
    try:
        data, r_curve, q_curve, market_options, base_name = load_latest_calibration()
        plot_surface_professional(data['market']['S0'], r_curve, q_curve, data.get('analytical', {}), "Asset", base_name, market_options, data)
    except Exception as e: print(f"Error: {e}")

if __name__ == "__main__": main()