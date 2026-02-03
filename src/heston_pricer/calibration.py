import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass
from .analytics import HestonAnalyticalPricer
from typing import List, Dict
from scipy.interpolate import interp1d
from collections import defaultdict
from .models.process import HestonProcess
from .market import MarketEnvironment
import warnings
from .models.mc_kernels import generate_heston_paths_crn

class HestonCalibrator:
    """
    Calibrates Heston model parameters to market option prices using 
    L-BFGS-B optimization with INVERSE SPREAD WEIGHTING.
    
    Weights options by 1/Spread to prioritize liquid, high-confidence data.
    """
    
    def __init__(self, S0, r_curve, q_curve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def calibrate(self, options):
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        # --- QUICK & ROBUST WEIGHTING ---
        # Weight = 1 / Spread. This mutes noisy wings with wide spreads.
        # We floor the spread at 0.01 to prevent division by zero or infinite weights.
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        weights = 1.0 / spreads

        # Standardized Order: kappa, theta, xi, rho, v0
        bounds = [(0.1, 8.0), (1e-4, 0.5), (0.01, 2.0), (-0.95, 0.0), (1e-4, 0.5)]
        x0 = [2.0, 0.04, 0.4, -0.7, 0.04]

        def objective(p):
            kappa, theta, xi, rho, v0 = p
            try:
                model_p = HestonAnalyticalPricer.price_european_call_vectorized(
                    self.S0, strikes, maturities, r_vec, q_vec, kappa, theta, xi, rho, v0
                )
                # Optimization is done on the weighted error
                weighted_diff = (model_p - market_prices) * weights
                return np.sqrt(np.mean(weighted_diff**2)) 
            except Exception:
                return 1e12

        def callback(xk):
            # Calculate the ACTUAL Price RMSE for clear logging
            model_p = HestonAnalyticalPricer.price_european_call_vectorized(
                self.S0, strikes, maturities, r_vec, q_vec, *xk
            )
            real_rmse = np.sqrt(np.mean((model_p - market_prices)**2))
            w_obj = objective(xk)
            print(f"   [Step] W-Obj: {w_obj:.4f} | RMSE ($): {real_rmse:.4f} | k:{xk[0]:.2f} th:{xk[1]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f} v0:{xk[2]:.3f}")

        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            callback=callback, 
            tol=1e-9, 
            options={'eps': 1e-3, 'maxiter': 500}
        )
        
        # Final Price RMSE calculation
        final_p = HestonAnalyticalPricer.price_european_call_vectorized(self.S0, strikes, maturities, r_vec, q_vec, *res.x)
        
        return {
            **dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0'], res.x)), 
            "weighted_obj": res.fun,
            "rmse": np.sqrt(np.mean((final_p - market_prices)**2))
        }
    
class HestonCalibratorMC:
    """
    Monte Carlo Calibrator using Vectorized CRN and Inverse Spread Weighting.
    """
    def __init__(self, S0, r_curve, q_curve, n_paths=20000, n_steps=250):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve
        self.n_paths, self.min_n_steps = n_paths, n_steps
        self.z_noise = None
        self.maturity_map = defaultdict(list)
        self.weights_map = defaultdict(list) 
        self.maturity_indices = {}
        self.T_max, self.dt = 0.0, 0.0

    def _precompute(self, options):
        self.maturity_map.clear()
        self.weights_map.clear()
        self.maturity_indices.clear()
        
        for opt in options:
            T = opt.maturity
            self.maturity_map[T].append(opt)
            
            # --- WEIGHT CALCULATION (Inverse Spread) ---
            spread = max(abs(opt.ask - opt.bid), 0.01)
            self.weights_map[T].append(1.0 / spread)
            
        self.T_max = max(self.maturity_map.keys())
        self.dt = self.T_max / self.min_n_steps
        for T in self.maturity_map:
            idx = int(round(T / self.dt))
            self.maturity_indices[T] = max(1, min(idx, self.min_n_steps))

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

    def calibrate(self, options):
        self._precompute(options)
        # Standardized Order: kappa, theta, xi, rho, v0
        x0 = [2.0, 0.04, 0.4, -0.7, 0.04]
        bounds = [(0.1, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.0), (0.001, 1.0)]

        def objective(p):
            # Feller condition penalty to keep MC stable
            feller_penalty = 0.0
            if 2 * p[0] * p[1] < p[2] ** 2:
                feller_penalty = 5.0 * (p[2]**2 - 2 * p[0] * p[1])
                
            mod, mkt, w = self.get_prices(p)
            return np.sqrt(np.mean(((mod - mkt) * w) ** 2)) + feller_penalty

        def callback(xk):
            mod, mkt, _ = self.get_prices(xk)
            real_rmse = np.sqrt(np.mean((mod - mkt)**2))
            w_obj = objective(xk)
            print(f"   [MC-Step] W-Obj: {w_obj:.4f} | RMSE ($): {real_rmse:.4f} | k:{xk[0]:.2f} th:{xk[1]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f} v0:{xk[2]:.3f}")

        res = minimize(
            objective, x0,
            method='SLSQP', bounds=bounds,
            callback=callback,
            tol=1e-9, 
            options={'eps': 1e-3, 'maxiter': 500}
        )
        
        # Final Price RMSE check
        f_mod, f_mkt, _ = self.get_prices(res.x)
        return {
            **dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0'], res.x)), 
            "weighted_obj": res.fun,
            "rmse": np.sqrt(np.mean((f_mod - f_mkt)**2))
        }