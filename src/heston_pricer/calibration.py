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
from .models.mc_kernels import generate_heston_paths_crn
class BatesCalibrator:
    """
    Calibrates Bates (Heston + Jumps) model parameters using L-BFGS-B optimization 
    with CAPPED INVERSE SPREAD WEIGHTING for robustness.
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

    def calibrate(self, options, sigma_cap=2.0):
        """
        Runs the Bates calibration routine.
        
        Order: kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
        """
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        types = np.array([o.option_type for o in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        weights = self._calculate_robust_weights(options, sigma_cap)

        # Order: kappa, theta, xi, rho, v0, lamb (intensity), mu_j (mean jump), sigma_j (jump vol)
        bounds = [
            (0.1, 8.0),   # kappa
            (1e-4, 0.5),  # theta
            (0.01, 2.0),  # xi
            (-0.95, 0.0), # rho
            (1e-4, 0.5),  # v0
            (0.0, 3.0),   # lamb (jumps per year)
            (-0.6, 0.1),  # mu_j (mean log jump - usually negative for equity)
            (0.01, 0.5)   # sigma_j (jump volatility)
        ]
        
        # Smart initial guess: Start with Heston defaults + small jumps
        x0 = [2.0, 0.04, 0.4, -0.7, 0.04, 0.1, -0.1, 0.1]

        def objective(p):
            try:
                # Unpack all 8 parameters
                kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p
                
                model_p = BatesAnalyticalPricer.price_vectorized(
                self.S0, strikes, maturities, r_vec, q_vec, types, # Pass types here
                kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                )
                
                # RMSE with robust weights
                weighted_diff = (model_p - market_prices) * weights
                return np.sqrt(np.mean(weighted_diff**2)) 
            except Exception:
                return 1e12

        def callback(xk):
            # Track calibration progress
            w_obj = objective(xk)
            print(f"   [Step] W-Obj: {w_obj:.4f} | "
                  f"k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} | "
                  f"L:{xk[5]:.2f} muJ:{xk[6]:.2f} sJ:{xk[7]:.2f}")

        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds, 
            callback=callback, 
            tol=1e-9, 
            options={'eps': 1e-3, 'maxiter': 500}
        )
        # Calculate final unweighted RMSE for reporting
        final_p = BatesAnalyticalPricer.price_vectorized(self.S0, strikes, maturities, r_vec, q_vec, types, *res.x)
        rmse = np.sqrt(np.mean((final_p - market_prices)**2))
        
        return {
            **dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x)), 
            "weighted_obj": res.fun,
            "rmse": rmse
        }
    
class HestonCalibratorMC:
    """
    Monte Carlo Calibrator using Vectorized CRN and Robust Inverse Spread Weighting.
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
        
        # 1. Calculate Robust Weights across ALL options first
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        raw_weights = 1.0 # / spreads
        cap_val = np.mean(raw_weights) + sigma_cap * np.std(raw_weights)
        all_weights = np.clip(raw_weights, None, cap_val)
        
        # 2. Map options and their corresponding robust weights
        for idx, opt in enumerate(options):
            T = opt.maturity
            self.maturity_map[T].append(opt)
            self.weights_map[T].append(all_weights[idx])
            
        self.T_max = max(self.maturity_map.keys())
        self.dt = self.T_max / self.min_n_steps
        for T in self.maturity_map:
            step_idx = int(round(T / self.dt))
            self.maturity_indices[T] = max(1, min(step_idx, self.min_n_steps))

        if self.z_noise is None or self.z_noise.shape[1] != self.min_n_steps:
            rng = np.random.default_rng(42)
            self.z_noise = rng.standard_normal((2, self.min_n_steps, self.n_paths))

    def get_prices(self, params):
        kappa, theta, xi, rho, v0 = params
        paths = generate_heston_paths_crn(
            self.S0, 0.0, 0.0, v0, kappa, theta, xi, rho,
            self.T_max, self.n_paths, self.min_n_steps, self.z_noise
        )
        model_prices, market_prices, weights = [], [], []
        for T, opts in self.maturity_map.items():
            time_idx = self.maturity_indices[T]
            S_T = paths[:, time_idx]
            strikes = np.array([o.strike for o in opts])
            
            r, q = self.r_curve.get_rate(T), self.q_curve.get_rate(T)
            F = self.S0 * np.exp((r - q) * T)
            S_T_adj = S_T * (F / self.S0)
            
            payoffs_adj = np.maximum(S_T_adj[None, :] - strikes[:, None], 0.0)
            vals = np.mean(payoffs_adj, axis=1) * np.exp(-r * T)
            
            model_prices.extend(vals)
            market_prices.extend([o.market_price for o in opts])
            weights.extend(self.weights_map[T])
            
        return np.array(model_prices), np.array(market_prices), np.array(weights)

    def calibrate(self, options, sigma_cap=2.0):
        self._precompute(options, sigma_cap)
        x0 = [2.0, 0.04, 0.4, -0.7, 0.04]
        bounds = [(0.1, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.0), (0.001, 1.0)]

        def objective(p):
            # Optional Feller penalty (disabled here by 0.0 coefficient, but structure preserved)
            feller_penalty = 0.0
            if 2 * p[0] * p[1] < p[2] ** 2:
                feller_penalty = 0.0 * (p[2]**2 - 2 * p[0] * p[1])
                
            mod, mkt, w = self.get_prices(p)
            return np.sqrt(np.mean(((mod - mkt) * w) ** 2)) + feller_penalty

        def callback(xk):
            mod, mkt, _ = self.get_prices(xk)
            real_rmse = np.sqrt(np.mean((mod - mkt)**2))
            w_obj = objective(xk)
            print(f"   [MC-Step] W-Obj: {w_obj:.4f} | RMSE ($): {real_rmse:.4f} | k:{xk[0]:.2f} th:{xk[1]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f} v0:{xk[4]:.3f}")

        res = minimize(
            objective, x0,
            method='SLSQP', bounds=bounds,
            callback=callback,
            tol=1e-9, 
            options={'eps': 1e-3, 'maxiter': 500}
        )
        
        f_mod, f_mkt, _ = self.get_prices(res.x)
        return {
            **dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0'], res.x)), 
            "weighted_obj": res.fun,
            "rmse": np.sqrt(np.mean((f_mod - f_mkt)**2))
        }