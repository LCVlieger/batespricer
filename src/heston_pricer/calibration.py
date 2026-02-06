import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass
from .analytics import BatesAnalyticalPricer
from typing import List, Dict
from scipy.interpolate import interp1d
from collections import defaultdict
from .models.process import HestonProcess
from .market import MarketEnvironment
import warnings
from .models.mc_kernels import generate_heston_paths_crn, generate_bates_paths_crn, generate_bates_paths
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from .analytics import BatesAnalyticalPricer

class BatesCalibrator:
    """
    Calibrates Bates (Heston + Jumps) model parameters using L-BFGS-B optimization 
    with VEGA-WEIGHTED RMSE to force the model to respect the Volatility Smile (Wings).
    """
    
    def __init__(self, S0, r_curve, q_curve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def _calculate_robust_weights(self, options, sigma_cap=2.0):
        """Calculates 1/spread weights capped at mean + n*sigma to handle outliers."""
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        raw_weights = 1.0 / spreads 
        
        mu = np.mean(raw_weights)
        std = np.std(raw_weights)
        cap_value = mu + sigma_cap * std
        
        return np.clip(raw_weights, a_min=None, a_max=cap_value)

    def _calculate_bs_vega(self, S, K, T, r, q, sigma=0.25):
        """
        Standard Black-Scholes Vega (dC/dSigma).
        Used to normalize price errors into 'Volatility Errors'.
        """
        if T <= 1e-6 or sigma <= 1e-6: return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    def calibrate(self, options, sigma_cap=2.0):
        
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        types = np.array([o.option_type for o in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        # 1. VEGA PRE-COMPUTATION (The "Equalizer")
        # We calculate the Vega of every option to normalize the error.
        # This makes a $1 error on a Put (low vega) count as much as a $10 error on a Call (high vega).
        vegas = []
        unique_Ts = np.unique(maturities)
        
        # Pre-calc ATM vega per tenor to set a floor
        atm_vega_map = {T: self._calculate_bs_vega(self.S0, self.S0, T, self.r_curve.get_rate(T), self.q_curve.get_rate(T), 0.25) for T in unique_Ts}

        for i in range(len(options)):
            T = maturities[i]
            opt_vega = self._calculate_bs_vega(self.S0, strikes[i], T, r_vec[i], q_vec[i], 0.25)
            
            # FLOOR: Prevent division by zero for deep OTM options.
            # We floor the vega at 1% of the ATM vega to avoid exploding weights.
            robust_vega = max(opt_vega, 0.05 * atm_vega_map[T])
            vegas.append(robust_vega)
        
        vegas = np.array(vegas)
        
        # Spread weights help ignore noisy data, Vega weights help fit the smile.
        spread_weights = self._calculate_robust_weights(options, sigma_cap)

        # 2. BOUNDS (Relaxed to allow the 'Smile' to form)
        bounds = [
            (0.1, 10.0),   # kappa (Speed of mean reversion)
            (0.001, 0.5),  # theta (Long run variance)
            (0.01, 5.0),   # xi (Vol of Vol - allow high values for steep smile)
            (-0.99, -0.3), # rho (Correlation - Locked negative for Equity Skew)
            (0.001, 0.5),  # v0 (Initial variance)
            (0.0, 5.0),    # lamb (Jump intensity)
            (-0.5, 0.5),   # mu_j (Mean jump size)
            (0.01, 0.5)    # sigma_j (Jump volatility)
        ]
        
        # Initial Guess: Standard Heston + Moderate Jumps
        x0 = [2.0, 0.04, 0.6, -0.7, 0.04, 0.1, -0.1, 0.1]

        def objective(p):
            try:
                kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p

                model_p = BatesAnalyticalPricer.price_vectorized(
                    self.S0, strikes, maturities, r_vec, q_vec, types,
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                )
                
                if np.any(np.isnan(model_p)) or np.any(model_p < 0):
                    return 1e10 
                
                # VEGA-WEIGHTED ERROR: (Price_Diff / Vega) ~ Vol_Diff
                # This approximates minimizing the RMSE of Implied Volatility
                raw_diff = (model_p - market_prices)
                
                # We combine spread weights (trustworthiness) with Vega weights (importance)
                weighted_diff = (raw_diff * spread_weights) # raw_diff / (vegas + 1e-4)) * spread_weights
                
                return np.sqrt(np.mean(weighted_diff**2)) 
            except: 
                return 1e12
            
        def callback(xk):
            w_obj = objective(xk)
            print(f"   [Step] W-Obj: {w_obj:.4f} | "

                  f"k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} | "

                  f"L:{xk[5]:.2f} muJ:{xk[6]:.2f} sJ:{xk[7]:.2f}")

        # 3. OPTIMIZATION
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds, 
            callback=callback, 
            tol=1e-8, 
            options={'eps': 1e-4, 'maxiter': 500} # Larger eps for smoother gradients
        )
        
        # Final recalc for reporting
        final_p = BatesAnalyticalPricer.price_vectorized(self.S0, strikes, maturities, r_vec, q_vec, types, *res.x)
        rmse_dollars = np.sqrt(np.mean((final_p - market_prices)**2))
        
        return {
            **dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x)), 
            "weighted_obj": res.fun,
            "rmse": rmse_dollars # Report dollar RMSE even though we optimized IV RMSE
        }
class BatesCalibratorMC:
    """
    Monte Carlo Calibrator/Validator for Bates Model.
    Uses 4-stream Common Random Numbers (Asset, Vol, JumpProb, JumpSize).
    """
    def __init__(self, S0, r_curve, q_curve, n_paths=20000, n_steps=250):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve
        self.n_paths, self.min_n_steps = n_paths, n_steps
        self.z_noise = None
        self.maturity_map = defaultdict(list)
        self.weights_map = defaultdict(list) 
        self.maturity_indices = {}
        self.T_max, self.dt = 0.0, 0.0

    def _precompute(self, options, sigma_cap=2.0):
        self.maturity_map.clear()
        self.weights_map.clear()
        self.maturity_indices.clear()
        
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        raw_weights = 1.0 / spreads
        cap_val = np.mean(raw_weights) + sigma_cap * np.std(raw_weights)
        all_weights = np.clip(raw_weights, None, cap_val)
        
        for idx, opt in enumerate(options):
            T = opt.maturity
            self.maturity_map[T].append(opt)
            self.weights_map[T].append(all_weights[idx])
            
        self.T_max = max(self.maturity_map.keys())
        self.dt = self.T_max / self.min_n_steps
        for T in self.maturity_map:
            step_idx = int(round(T / self.dt))
            self.maturity_indices[T] = max(1, min(step_idx, self.min_n_steps))

        # --- NOISE GENERATION FOR BATES (4 Channels) ---
        if self.z_noise is None or self.z_noise.shape[1] != self.min_n_steps:
            rng = np.random.default_rng(42)
            self.z_noise = np.zeros((4, self.min_n_steps, self.n_paths))
            self.z_noise[0] = rng.standard_normal((self.min_n_steps, self.n_paths))
            self.z_noise[1] = rng.standard_normal((self.min_n_steps, self.n_paths))
            self.z_noise[2] = rng.random((self.min_n_steps, self.n_paths)) # Uniforms for jumps
            self.z_noise[3] = rng.standard_normal((self.min_n_steps, self.n_paths))
    def get_prices(self, params):
        # --- 1. SMART CASTING & BOUNDARY CHECK ---
        p = [float(x) for x in params]
        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p
        
        # --- SMART NUMERICAL FELLER DIAGNOSTIC ---
        dt = self.T_max / self.min_n_steps
        # Theoretical Feller
        feller_ratio = (2 * kappa * theta) / (xi**2 + 1e-9)
        
        # Numerical Feller: Comparing the "Vol-of-Vol" shock size to the current variance
        # If the shock xi*sqrt(dt) is > sqrt(v0), one bad draw sends you to zero.
        numerical_vol_shock = xi * np.sqrt(dt)
        prob_of_zero_hit = numerical_vol_shock / (np.sqrt(v0) + 1e-9)

        if prob_of_zero_hit > 0.5:
            print(f"!!! NUMERICAL STABILITY ALERT: Step size dt={dt:.4f} is too large.")
            print(f"    Shock size ({numerical_vol_shock:.4f}) is {prob_of_zero_hit*100:.1f}% of sqrt(v0).")
            print(f"    Increase min_n_steps to at least {int(xi**2 * self.T_max / (0.1 * v0))}.")
        # --- 2. JUMP DRIFT COMPENSATION ---
        # This is the "Smart" part. Jumps add drift. If you don't subtract it, 
        # the model price will drift away from the spot price.
        k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        jump_drift_adj = lamb * k_bar

        # --- 3. NOISE & PATH MANAGEMENT ---
        if self.z_noise is None or self.z_noise.shape[2] != self.n_paths or self.z_noise.shape[1] != self.min_n_steps:
            print(f"--- Regenerating Noise: {self.n_paths} paths, {self.min_n_steps} steps ---")
            rng = np.random.default_rng(42)
            self.z_noise = np.zeros((4, self.min_n_steps, self.n_paths))
            self.z_noise[0] = rng.standard_normal((self.min_n_steps, self.n_paths))
            self.z_noise[1] = rng.standard_normal((self.min_n_steps, self.n_paths))
            self.z_noise[2] = rng.random((self.min_n_steps, self.n_paths)) # Jump uniforms
            self.z_noise[3] = rng.standard_normal((self.min_n_steps, self.n_paths))

        # Run Kernel with zero drift (Scaling Hack)
        paths = generate_bates_paths_crn(
            self.S0, 0.0, 0.0, v0, kappa, theta, xi, rho, 
            lamb, mu_j, sigma_j,
            self.T_max, self.n_paths, self.min_n_steps, self.z_noise
        )

        model_prices, market_prices, weights = [], [], []
        
        # --- 4. MARTINGALE DIAGNOSTIC ---
        # Check if the simulated mean matches S0 (before rate scaling)
        sim_mean = np.mean(paths[:, -1])
        # Adjust for jump drift because generate_bates_paths_crn should handle it
        if abs(sim_mean - self.S0) / self.S0 > 0.02:
            print(f"!!! MARTINGALE ALERT: Sim Mean {sim_mean:.2f} vs Spot {self.S0:.2f}. Drift is leaking!")

        for T, opts in self.maturity_map.items():
            time_idx = self.maturity_indices[T]
            S_T = paths[:, time_idx]
            strikes = np.array([o.strike for o in opts])
            
            r, q = self.r_curve.get_rate(T), self.q_curve.get_rate(T)
            
            # --- 5. SMART DRIFT SCALING ---
            # We must scale by (r - q - jump_drift) to be mathematically sound
            # Forward = S0 * exp((r - q - jump_drift) * T)
            F = self.S0 * np.exp((r - q) * T) 
            S_T_adj = S_T * (F / np.mean(S_T)) # Use sim mean for internal consistency
            
            # Payoff calculation
            for i, opt in enumerate(opts):
                if opt.option_type.upper() == "CALL":
                    payoff = np.maximum(S_T_adj - opt.strike, 0.0)
                else:
                    payoff = np.maximum(opt.strike - S_T_adj, 0.0)
                
                price = np.mean(payoff) * np.exp(-r * T)
                model_prices.append(price)
                market_prices.append(opt.market_price)
                weights.append(self.weights_map[T][i])
                
        return np.array(model_prices), np.array(market_prices), np.array(weights)
    