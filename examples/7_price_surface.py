import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import glob
import os
from matplotlib.transforms import Bbox

def visualize_price_surface_final(csv_file):
    """
    Final Polish: Triangular Surface with full Bates parameters,
    strict font consistency, lowered tilt, and 'inspiration-style' layout.
    """
    print(f"Generating Final Polished Plot from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 1. Load Metadata & Bates Parameters
    base_name = csv_file.replace("_prices.csv", "")
    json_file = f"{base_name}_meta.json"
    
    # Default values
    s0 = df['K'].median()
    params = {}
    ticker = "Asset"
    
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

    df['Moneyness'] = df['K'] / s0

    # 2. Setup Plot (White Background)
    #plt.style.use('default')
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_facecolor('white')

    # --- Layer A: Triangular Surface ---
    surf = ax.plot_trisurf(
        df['Moneyness'], df['T'], df['Model'], 
        cmap=cm.viridis, 
        alpha=0.6, 
        edgecolor='none', 
        linewidth=0, 
        antialiased=True,
    )
    
    # --- Layer B: Market Dots ---
    #ax.scatter(
    #    df['Moneyness'], df['T'], df['Market'], 
    #    color='r', #'#D72638'
    #    s=15, 
    #    alpha=0.9,
    #    depthshade=False,
    #    label='Market Price' # Label for legend
    #)
    ax.scatter(
        df['Moneyness'], df['T'], df['Market'], 
        color='r', 
        s=13, 
        label='Market Prices', 
        depthshade=False
    )

    # --- Layer C: Needles ---
    for x, y, z_mkt, z_mod in zip(df['Moneyness'], df['T'], df['Market'], df['Model']):
        ax.plot([x, x], [y, y], [z_mkt, z_mod], color='black', alpha=0.3, linewidth=0.8)

    # 3. LEGEND (Exact Positioning from Inspiration)
    # bbox_to_anchor=(0.157, 0.797)
    ax.legend(
        loc='upper left', 
        bbox_to_anchor=(0.157, 0.797), 
        frameon=False, 
        labelcolor="black", 
        fontsize=10,
    )

    # 4. TITLES (With FULL Bates Parameters)
    # Extract params
    kappa = params.get('kappa', 0.0)
    theta = params.get('theta', 0.0)
    xi = params.get('xi', 0.0)
    rho = params.get('rho', 0.0)
    v0 = params.get('v0', 0.0)
    lamb = params.get('lamb', 0.0)
    mu_j = params.get('mu_j', 0.0)
    sigma_j = params.get('sigma_j', 0.0)

    # Main Title (Coordinate: 0.535, 0.84)
    fig.text(0.5975, 0.85, f"Bates Calibration Price Surface: {ticker}", 
             fontsize=16, fontweight='bold', family='monospace', ha='center', color='black')
    
    # Subtitle (Two lines to fit all parameters + S0)
    # Coordinate: 0.535, 0.79
    subtitle = (rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$" + "\n" +
                rf"$\lambda={lamb:.2f}, \mu_J={mu_j:.2f}, \sigma_J={sigma_j:.2f}, S_0={s0:.1f}$")
    
    fig.text(0.5975, 0.80, subtitle, fontsize=9, family='monospace', ha='center', color='#555555')

    # 5. STYLING & FONTS
    # Transparent panes
    #ax.xaxis.set_pane_color((1, 1, 1, 0))
    #ax.yaxis.set_pane_color((1, 1, 1, 0))
    #ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.dist = 14
    # Gray dotted grid (from inspiration)
    #ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
    #ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
    #ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
    #ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    # Labels
    ax.set_xlabel('Moneyness ($K/S_0$)', labelpad=10)
    ax.set_ylabel('Maturity ($T$ Years)', labelpad=10)
    ax.set_zlabel('Option Price ($)', labelpad=10)

    # --- CRITICAL FIX: Unify Axis Tick Fonts ---
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update({
            'linewidth': 0.575,              # Much thinner than default
            'alpha':1
        })
        for label in axis.get_ticklabels():
            label.set_family('monospace')

    # TILT: Lowered to 22 (from 30)
    #ax.view_init(elev=19.7, azim=-119)
    ax.view_init(elev=30, azim=-120)

    # Tight layout logic
    fig.subplots_adjust(left=0.25, bottom=0.05, right=0.95, top=0.85)

    # 6. Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('monospace')

    # 7. Save
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    bbox_shifted = Bbox([[bbox.x0 - 0.15, bbox.y0], [bbox.x1+0.075, bbox.y1+0.15]])
    save_path = csv_file.replace(".csv", "_price_surface_final.png")
    plt.savefig(save_path, dpi=300, bbox_inches=bbox_shifted)
    print(f"-> Saved: {save_path}")

# --- Run ---
files = glob.glob("results/calibration_*_prices.csv")
if files:
    # Pick the latest file
    latest_file = max(files, key=os.path.getctime)
    visualize_price_surface_final(latest_file)
else:
    print("No calibration files found.")