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
            (-0.99, 0.0), # rho (Correlation - Locked negative for Equity Skew)
            (0.001, 0.5),  # v0 (Initial variance)
            (0.0, 5.0),    # lamb (Jump intensity)
            (-0.5, 0.5),   # mu_j (Mean jump size)
            (0.01, 0.5)    # sigma_j (Jump volatility)
        ]
        
        # Initial Guess: Standard Heston + Moderate Jumps
        x0 = [1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]#[2.0, 0.04, 0.6, -0.7, 0.04, 0.1, -0.1, 0.1] #[3.2, 0.040, 0.82, -0.74, 0.022, 0.0376, -0.47, 0.37] #
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
                spreads2 = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
                # We combine spread weights (trustworthiness) with Vega weights (importance)
                lambda_floor = 0.05 * np.mean(list(atm_vega_map.values()))
                weighted_diff = (raw_diff * spread_weights) #weighted_diff = raw_diff / (vegas + lambda_floor) #+ spreads2)#(raw_diff * spread_weights) ## raw_diff / (vegas + 1e-4)) * spread_weights
                
                return np.sqrt(np.mean(weighted_diff**2)) 
            except: 
                return 1e12
            
        def callback(xk):
            w_obj = objective(xk)
            print(f"   [Step] W-Obj: {w_obj:.4f} | "

                  f"k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f}| "

                  f"L:{xk[5]:.2f} muJ:{xk[6]:.2f} sJ:{xk[7]:.2f}")

        # 3. OPTIMIZATION
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds, 
            callback=callback, 
            tol=1e-8, ######1e-8
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
import numpy as np
from numba import njit, prange
from collections import defaultdict

@njit(parallel=True, fastmath=True)
def _numba_price_engine(paths, time_idxs, strikes, is_calls, rates, qs, drift_corr, maturities):
    """
    Optimized pricing engine. prange parallelizes across options.
    """
    n_opts = len(strikes)
    n_paths = paths.shape[0]
    prices = np.zeros(n_opts)
    
    for i in prange(n_opts):
        t_idx = time_idxs[i]
        K = strikes[i]
        T = maturities[i]
        r = rates[i]
        q = qs[i]
        is_call = is_calls[i]
        
        # Apply the exact drift adjustment from your original logic
        adj_factor = np.exp((r - q - drift_corr) * T)
        disc = np.exp(-r * T)
        
        payoff_sum = 0.0
        for p in range(n_paths):
            # paths[:, time_idx] access
            S_final = paths[p, t_idx] * adj_factor
            
            if is_call:
                val = S_final - K
            else:
                val = K - S_final
            
            if val > 0:
                payoff_sum += val
        
        prices[i] = (payoff_sum / n_paths) * disc
        
    return prices

class BatesCalibratorMC:
    def __init__(self, S0, r_curve, q_curve, n_paths=20000, n_steps=250):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve
        self.n_paths, self.min_n_steps = n_paths, n_steps
        self.z_noise = None
        
        # Internal state for flattened arrays
        self.f_strikes = None
        self.f_is_call = None
        self.f_time_idxs = None
        self.f_rates = None
        self.f_qs = None
        self.f_maturities = None
        self.f_market_prices = None
        self.f_weights = None

    def _precompute(self, options, sigma_cap=2.0):
        # 1. Setup Time Grid
        self.T_max = max(o.maturity for o in options)
        self.dt = self.T_max / self.min_n_steps
        
        # Ensure even number of paths for Antithetic Variates
        if self.n_paths % 2 != 0:
            self.n_paths += 1
            
        # 2. Weights Logic (Unchanged)
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        raw_weights = 1.0 / spreads
        cap_val = np.mean(raw_weights) + sigma_cap * np.std(raw_weights)
        all_weights = np.clip(raw_weights, None, cap_val)
        
        # 3. Flatten Arrays (Unchanged)
        self.f_strikes = np.array([o.strike for o in options], dtype=np.float64)
        self.f_is_call = np.array([o.option_type.upper() == 'CALL' for o in options], dtype=np.bool_)
        self.f_maturities = np.array([o.maturity for o in options], dtype=np.float64)
        self.f_market_prices = np.array([o.market_price for o in options], dtype=np.float64)
        self.f_weights = all_weights.astype(np.float64)
        self.f_rates = np.array([self.r_curve.get_rate(o.maturity) for o in options])
        self.f_qs = np.array([self.q_curve.get_rate(o.maturity) for o in options])
        self.f_time_idxs = np.array([max(1, min(int(round(o.maturity / self.dt)), self.min_n_steps)) 
                                    for o in options], dtype=np.int32)

        # 4. Antithetic Noise Generation (CORRECTED)
        if (self.z_noise is None or self.z_noise.shape[1] != self.min_n_steps or self.z_noise.shape[2] != self.n_paths):
            rng = np.random.default_rng(42)
            half_paths = self.n_paths // 2
            
            # Create Full Noise Matrix
            self.z_noise = np.zeros((4, self.min_n_steps, self.n_paths))
            
            # --- Generate First Half ---
            z_half_1 = rng.standard_normal((3, self.min_n_steps, half_paths)) # Rows 0 (Asset), 1 (Vol), 3 (JumpSize)
            u_half_1 = rng.random((self.min_n_steps, half_paths))             # Row 2 (Poisson Trigger)
            
            self.z_noise[0, :, :half_paths] = z_half_1[0]
            self.z_noise[1, :, :half_paths] = z_half_1[1]
            self.z_noise[3, :, :half_paths] = z_half_1[2]
            self.z_noise[2, :, :half_paths] = u_half_1

            # --- Generate Second Half (Antithetic Mixed) ---
            
            # 1. Brownian Motions: Use Antithetic (Invert Signs) -> Reduces Variance
            self.z_noise[0, :, half_paths:] = -z_half_1[0]
            self.z_noise[1, :, half_paths:] = -z_half_1[1]
            self.z_noise[3, :, half_paths:] = -z_half_1[2] # Jump Size is Normal, so we can invert this too
            
            # 2. Poisson Triggers: INDEPENDENT -> Preserves Jump Statistics
            # DO NOT use (1-U) here. Generate fresh randoms.
            self.z_noise[2, :, half_paths:] = rng.random((self.min_n_steps, half_paths))
    def get_prices(self, params):
            kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = params
            
            # 1. Generate Paths (Risk Neutral, r=0, q=0)
            # Result shape: [n_paths, n_steps + 1]
            paths = generate_bates_paths_crn(
                self.S0, 0.0, 0.0, v0, kappa, theta, xi, rho, 
                lamb, mu_j, sigma_j,
                self.T_max, self.n_paths, self.min_n_steps, self.z_noise
            )
            avg_terminal = np.mean(paths[:, -1])

            ratio = avg_terminal / self.S0
            if abs(ratio - 1.0) > 0.01: print(f"Error: {ratio}")
            # =======================================================
            # VECTORIZED MARTINGALE CORRECTION
            # =======================================================
            # Instead of checking just the last step, we check the average
            # of the paths at EVERY time step (axis=0).
            # This creates a vector of averages with shape (n_steps + 1,)
            avg_paths = np.mean(paths, axis=0)
            
            # Calculate a correction factor for every single time step.
            # This forces the mean at t=1, t=50, t=100... all to equal S0 exactly.
            # We avoid division by zero by replacing 0s (unlikely) with 1s if needed, 
            # though standard paths shouldn't hit 0 mean.
            corrections = self.S0 / avg_paths
            # Apply correction via Broadcasting.
            # Numpy matches the last dimension: 
            # (n_paths, n_steps+1) * (n_steps+1,) -> Scales each column correctly.
            paths *= corrections
            
            # DEBUG: Verify the correction works (Optional)
            # new_avgs = np.mean(paths, axis=0)
            # if np.max(np.abs(new_avgs - self.S0)) > 1e-9:
            #    print("Drift correction failed!")
            # =======================================================
            
            #print(f" correction + {ratio:.5f}")
            # 3. Pricing Engine
            # Now pass the corrected paths. Since we forced the mean to S0,
            # the drift_corr parameter in the engine stays 0.0.
            model_prices = _numba_price_engine(
                paths, self.f_time_idxs, self.f_strikes, self.f_is_call, 
                self.f_rates, self.f_qs, 0.0, self.f_maturities
            )
            
            return model_prices, self.f_market_prices, self.f_weights

        
                