import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
from numba import njit, prange
from .analytics import BatesAnalyticalPricer, BatesAnalyticalPricerFast
from .models.mc_kernels import generate_bates_paths_crn, generate_bates_qe_slices_crn

@njit(parallel=True, fastmath=True)
def _numba_price_engine(paths, time_idxs, strikes, is_calls, rates, qs, drift_corr, maturities):
    n_opts, n_paths = len(strikes), paths.shape[0]
    prices = np.zeros(n_opts)
    
    for i in prange(n_opts):
        t_idx, K, T, r, q, is_call = time_idxs[i], strikes[i], maturities[i], rates[i], qs[i], is_calls[i]
        adj = np.exp((r - q - drift_corr) * T)
        disc = np.exp(-r * T)
        
        payoff_sum = 0.0
        for p in range(n_paths):
            s_final = paths[p, t_idx] * adj
            val = s_final - K if is_call else K - s_final
            if val > 0:
                payoff_sum += val
        prices[i] = (payoff_sum / n_paths) * disc
    return prices

class BatesCalibrator:
    def __init__(self, S0: float, r_curve, q_curve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def _get_weights(self, options, sigma_cap=2.0):
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        w = 1.0 / spreads 
        return np.clip(w, None, np.mean(w) + sigma_cap * np.std(w))

    def _bs_vega(self, S, K, T, r, q, sigma=0.25):
        if T <= 1e-6 or sigma <= 1e-6: return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    def calibrate(self, options: List, sigma_cap=2.0, model="Bates") -> Dict:
        strikes = np.array([o.strike for o in options])
        mats = np.array([o.maturity for o in options])
        mkt_p = np.array([o.market_price for o in options])
        types = np.array([o.option_type for o in options])
        rv, qv = np.array([self.r_curve.get_rate(t) for t in mats]), np.array([self.q_curve.get_rate(t) for t in mats])
        
        atm_vegas = {T: self._bs_vega(self.S0, self.S0, T, self.r_curve.get_rate(T), self.q_curve.get_rate(T)) for T in np.unique(mats)}
        #vegas = np.array([max(self._bs_vega(self.S0, s, t, r, q), 0.05 * atm_vega_map[t]) for s, t, r, q in zip(strikes, mats, rv, qv)])
        weights = self._get_weights(options, sigma_cap)

        if model == "Bates":
            x0 = [1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]
            bnds = [(0.1, 10.), (0.001, 0.5), (0.01, 5.0), (-0.99, 0.0), (0.001, 0.5), (0.0, 5.0), (-0.5, 0.5), (0.01, 0.5)]
        elif model == "Heston":
            x0 = [1.5, 0.25, 0.6, -0.2, 0.21]
            bnds = [(0.1, 10.), (0.001, 0.5), (0.01, 5.0), (-0.99, 0.0), (0.001, 0.5)]
        elif model == "BS":
            x0 = [0.21]
            bnds = [(0.001, 0.5)]

        def obj(p):
            try:
                if model == "Bates": full_p = p
                elif model == "Heston": full_p = [p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, 0.0]
                elif model == "BS": full_p = [1e-4, p[0], 1e-4, 0.0, p[0], 0.0, 0.0, 0.0]
                mod_p = BatesAnalyticalPricer.price_vectorized(self.S0, strikes, mats, rv, qv, types, *full_p)
                if np.any(np.isnan(mod_p)) or np.any(mod_p < 0): return 1e10
                return np.sqrt(np.mean(((mod_p - mkt_p) * weights)**2))
            except: return 1e12

        def callback(xk):
            w_obj = obj(xk)
            if model == "Bates":
                print(f"   [Step] W-Obj: {w_obj:.4f} | k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f} | L:{xk[5]:.2f} muJ:{xk[6]:.2f} sJ:{xk[7]:.2f}")
            elif model == "Heston":
                print(f"   [Step] W-Obj: {w_obj:.4f} | k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f}")
            elif model == "BS":
                print(f"   [Step] W-Obj: {w_obj:.4f} | v0:{xk[0]:.4f}")

        res = minimize(obj, x0, method='L-BFGS-B', bounds=bnds, callback=callback, tol=1e-8, options={'eps': 1e-4, 'maxiter': 500})
        
        if model == "Bates": final_prms = res.x
        elif model == "Heston": final_prms = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 0.0, 0.0, 0.0]
        elif model == "BS": final_prms = [1e-4, res.x[0], 1e-4, 0.0, res.x[0], 0.0, 0.0, 0.0]

        final_p = BatesAnalyticalPricer.price_vectorized(self.S0, strikes, mats, rv, qv, types, *final_prms)
        
        return {**dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], final_prms)), 
                "weighted_obj": res.fun, "rmse": np.sqrt(np.mean((final_p - mkt_p)**2))}
    
class BatesCalibratorFast:
    """
    Calibrates Bates (Heston + Jumps) model parameters using L-BFGS-B optimization.
    Wired to use the ultra-fast Cached Gauss-Legendre pricer.
    """
    
    def __init__(self, S0, r_curve, q_curve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def _calculate_robust_weights(self, options, sigma_cap=2.0):
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        raw_weights = 1.0 / spreads 
        
        mu = np.mean(raw_weights)
        std = np.std(raw_weights)
        cap_value = mu + sigma_cap * std
        
        return np.clip(raw_weights, a_min=None, a_max=cap_value)

    def _calculate_bs_vega(self, S, K, T, r, q, sigma=0.25):
        if T <= 1e-6 or sigma <= 1e-6: return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    def calibrate(self, options, sigma_cap=2.0, model="Bates"):
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        types = np.array([o.option_type for o in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        vegas = []
        unique_Ts = np.unique(maturities)
        atm_vega_map = {T: self._calculate_bs_vega(self.S0, self.S0, T, self.r_curve.get_rate(T), self.q_curve.get_rate(T), 0.25) for T in unique_Ts}

        for i in range(len(options)):
            T = maturities[i]
            opt_vega = self._calculate_bs_vega(self.S0, strikes[i], T, r_vec[i], q_vec[i], 0.25)
            robust_vega = max(opt_vega, 0.05 * atm_vega_map[T])
            vegas.append(robust_vega)
        
        vegas = np.array(vegas)
        spread_weights = self._calculate_robust_weights(options, sigma_cap)

        if model == "Bates":
            bounds = [(0.1, 10.0), (0.001, 0.5), (0.01, 5.0), (-0.99, 0.0), (0.001, 0.5), (0.0, 5.0), (-0.5, 0.5), (0.01, 0.5)]
            x0 = [1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]
        elif model == "Heston":
            bounds = [(0.1, 10.0), (0.001, 0.5), (0.01, 5.0), (-0.99, 0.0), (0.001, 0.5)]
            x0 = [1.5, 0.25, 0.6, -0.2, 0.21]
        elif model == "BS":
            bounds = [(0.001, 0.5)]
            x0 = [0.21]
        
        def objective(p):
            try:
                if model == "Bates":
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p
                elif model == "Heston":
                    kappa, theta, xi, rho, v0 = p
                    lamb, mu_j, sigma_j = 0.0, 0.0, 0.0
                elif model == "BS":
                    v0 = p[0]
                    kappa, theta, xi, rho, lamb, mu_j, sigma_j = 1e-4, v0, 1e-4, 0.0, 0.0, 0.0, 0.0

                model_p = BatesAnalyticalPricerFast.price_vectorized(
                    self.S0, strikes, maturities, r_vec, q_vec, types,
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True
                )
                
                if np.any(np.isnan(model_p)) or np.any(model_p < 0):
                    return 1e10 
                raw_diff = (model_p - market_prices)
                weighted_diff = (raw_diff * spread_weights) 
                
                return np.sqrt(np.mean(weighted_diff**2)) 
            except: 
                return 1e12
            
        def callback(xk):
            w_obj = objective(xk)
            if model == "Bates":
                print(f"   [Step] W-Obj: {w_obj:.4f} | "
                      f"k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f}| "
                      f"L:{xk[5]:.2f} muJ:{xk[6]:.2f} sJ:{xk[7]:.2f}")
            elif model == "Heston":
                print(f"   [Step] W-Obj: {w_obj:.4f} | "
                      f"k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f}")
            elif model == "BS":
                print(f"   [Step] W-Obj: {w_obj:.4f} | v0:{xk[0]:.4f}")

        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds, 
            callback=callback, 
            tol=1e-8, 
            options={'eps': 1e-4, 'maxiter': 500} 
        )

        if model == "Bates": final_prms = res.x
        elif model == "Heston": final_prms = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 0.0, 0.0, 0.0]
        elif model == "BS": final_prms = [1e-4, res.x[0], 1e-4, 0.0, res.x[0], 0.0, 0.0, 0.0]

        final_p = BatesAnalyticalPricerFast.price_vectorized(self.S0, strikes, maturities, r_vec, q_vec, types, *final_prms, silent=True)
        rmse_dollars = np.sqrt(np.mean((final_p - market_prices)**2))
        
        return {
            **dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], final_prms)), 
            "weighted_obj": res.fun,
            "rmse": rmse_dollars 
        }
    
class BatesCalibratorMC:
    def __init__(self, S0, r_curve, q_curve, n_paths=20000, n_steps=250):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve
        self.n_paths, self.n_steps = n_paths + (n_paths % 2), n_steps
        self.z_noise = None

    def _precompute(self, options, sigma_cap=2.0):
        self.T_max = max(o.maturity for o in options)
        self.dt = self.T_max / self.n_steps
        
        spreads = np.array([max(abs(o.ask - o.bid), 0.01) for o in options])
        w = 1.0 / spreads
        self.f_weights = np.clip(w, None, np.mean(w) + sigma_cap * np.std(w)).astype(np.float64)
        
        self.f_strikes = np.array([o.strike for o in options])
        self.f_is_call = np.array([o.option_type.upper() == 'CALL' for o in options])
        self.f_maturities = np.array([o.maturity for o in options])
        self.f_market_prices = np.array([o.market_price for o in options])
        self.f_rates = np.array([self.r_curve.get_rate(o.maturity) for o in options])
        self.f_qs = np.array([self.q_curve.get_rate(o.maturity) for o in options])
        self.f_t_idxs = np.array([max(1, min(int(round(o.maturity / self.dt)), self.n_steps)) for o in options], dtype=np.int32)

        if self.z_noise is None:
            rng, hp = np.random.default_rng(42), self.n_paths // 2
            zn = np.zeros((4, self.n_steps, self.n_paths))
            z_gauss = rng.standard_normal((3, self.n_steps, hp))
            zn[[0,1,3], :, :hp] = z_gauss
            zn[2, :, :hp] = rng.random((self.n_steps, hp))
            zn[[0,1,3], :, hp:] = -z_gauss
            zn[2, :, hp:] = rng.random((self.n_steps, hp))
            self.z_noise = zn

    def get_prices(self, params):
        paths = generate_bates_paths_crn(self.S0, 0.0, 0.0, params[4], params[0], params[1], params[2], params[3], 
                                         params[5], params[6], params[7], self.T_max, self.n_paths, self.n_steps, self.z_noise)
        paths *= (self.S0 / np.maximum(np.mean(paths, axis=0), 1e-12))
        mod_p = _numba_price_engine(paths, self.f_t_idxs, self.f_strikes, self.f_is_call, self.f_rates, self.f_qs, 0.0, self.f_maturities)
        return mod_p, self.f_market_prices, self.f_weights

class BatesCalibratorMCFast(BatesCalibratorMC):
    def __init__(self, S0, r_curve, q_curve, n_paths=5000, n_steps_per_year=365):
        super().__init__(S0, r_curve, q_curve, n_paths, n_steps_per_year)

    def _precompute(self, options, sigma_cap=2.0):
        mats = np.sort(np.unique([o.maturity for o in options]))
        self.T_max = mats[-1]
        self.total_steps = max(1, int(round(self.T_max * self.n_steps)))
        self.dt = self.T_max / self.total_steps
        self.mat_idxs = np.array([max(1, int(round(T / self.dt))) for T in mats], dtype=np.int32)
        self.opt_map = np.array([np.where(mats == o.maturity)[0][0] for o in options], dtype=np.int32)
        
        super()._precompute(options, sigma_cap)
        if self.z_noise.shape[0] != 5:
            rng, hp = np.random.default_rng(42), self.n_paths // 2
            zn = np.zeros((5, self.total_steps, self.n_paths))
            z_g, u_g = rng.standard_normal((3, self.total_steps, hp)), rng.random((2, self.total_steps, hp))
            zn[:3, :, :hp], zn[3:, :, :hp] = z_g, u_g
            zn[:3, :, hp:], zn[3:, :, hp:] = -z_g, rng.random((2, self.total_steps, hp))
            self.z_noise = zn

    def get_prices(self, params):
        ka, th, xi, rho, v0, la, mj, sj = params
        slices = generate_bates_qe_slices_crn(self.S0, v0, ka, th, xi, rho, la, mj, sj, self.dt, self.n_paths, self.total_steps, self.mat_idxs, self.z_noise)
        slices *= (self.S0 / np.maximum(np.mean(slices, axis=0), 1e-12))
        
        st, dr = slices[:, self.opt_map], np.exp((self.f_rates - self.f_qs) * self.f_maturities)
        payoffs = np.where(self.f_is_call, np.maximum(st * dr - self.f_strikes, 0.), np.maximum(self.f_strikes - st * dr, 0.))
        return np.mean(payoffs * np.exp(-self.f_rates * self.f_maturities), axis=0), self.f_market_prices, self.f_weights

    def calibrate(self, options, sigma_cap=2.0, model="Bates"):
        self._precompute(options, sigma_cap)
        
        if model == "Bates":
            x0 = [1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]
            bnds = [(1.0, 5.0), (0.001, 0.5), (0.01, 0.9), (-0.99, 0.0), (0.001, 0.1), (0.0, 0.5), (-0.3, 0.0), (0.05, 0.3)]
        elif model == "Heston":
            x0 = [1.5, 0.25, 0.6, -0.2, 0.21]
            bnds = [(1.0, 5.0), (0.001, 0.5), (0.01, 0.9), (-0.99, 0.0), (0.001, 0.1)]
        elif model == "BS":
            x0 = [0.21]
            bnds = [(0.001, 0.1)]
        
        def obj(p):
            try:
                if model == "Bates": p_eval = p
                elif model == "Heston": p_eval = [p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, 0.0]
                elif model == "BS": p_eval = [1e-4, p[0], 1e-4, 0.0, p[0], 0.0, 0.0, 0.0]
                mp, mkp, w = self.get_prices(p_eval)
                if np.any(np.isnan(mp)) or np.any(mp < 0): return 1e10
                return np.sqrt(np.mean(((mp - mkp) * w)**2))
            except: return 1e12

        def callback(xk):
            w_obj = obj(xk)
            if model == "Bates":
                print(f"   [Step] W-Obj: {w_obj:.4f} | k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f} | L:{xk[5]:.2f} muJ:{xk[6]:.2f} sJ:{xk[7]:.2f}")
            elif model == "Heston":
                print(f"   [Step] W-Obj: {w_obj:.4f} | k:{xk[0]:.1f} th:{xk[1]:.3f} xi:{xk[2]:.2f} rho:{xk[3]:.2f} v0:{xk[4]:.2f}")
            elif model == "BS":
                print(f"   [Step] W-Obj: {w_obj:.4f} | v0:{xk[0]:.4f}")

        res = minimize(obj, x0, method='L-BFGS-B', bounds=bnds, callback=callback, tol=1e-8, options={'eps': 1e-3, 'maxiter': 250})
        
        if model == "Bates": final_prms = res.x
        elif model == "Heston": final_prms = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 0.0, 0.0, 0.0]
        elif model == "BS": final_prms = [1e-4, res.x[0], 1e-4, 0.0, res.x[0], 0.0, 0.0, 0.0]
        
        final_p, _, _ = self.get_prices(final_prms)
        return {**dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], final_prms)), 
                "weighted_obj": res.fun, "rmse": float(np.sqrt(np.mean((final_p - self.f_market_prices)**2)))}