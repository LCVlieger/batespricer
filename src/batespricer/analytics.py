import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import numpy.polynomial.legendre as leg

def implied_volatility(price, S, K, T, r, q, option_type="CALL"): 
    if price <= 0: return 0.0
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    intrinsic = max(K * df_r - S * df_q, 0) if option_type == "PUT" else max(S * df_q - K * df_r, 0)
    if price < intrinsic: return 0.0

    def bs_err(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "PUT":
            return (K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)) - price
        return (S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)) - price
    
    try: return brentq(bs_err, 1e-4, 5.0)
    except: return 0.0

class BatesAnalyticalPricer:
    @staticmethod
    def price_vectorized(S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=False):
        N_grid, u_limit = 296, 10000
        u = (np.linspace(0, 1, N_grid + 1)**2 * u_limit)[:, np.newaxis]
        u[0] = 1e-12 
        du = np.diff(u, axis=0) 
        
        K, T, r, q = map(np.atleast_1d, [K, T, r, q])
        T_m, r_m, q_m, K_m = T[None,:], r[None,:], q[None,:], K[None,:]
        F_m = S0 * np.exp((r_m - q_m) * T_m)

        def get_cf(phi):
            xs = np.maximum(xi, 1e-6)
            d = np.sqrt((kappa - rho * xs * phi * 1j)**2 + xs**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xs * phi * 1j - d) / (kappa - rho * xs * phi * 1j + d)
            et = np.exp(-d * T_m)
            
            D = ((kappa - rho * xs * phi * 1j - d) / xs**2) * ((1 - et) / (1 - g * et))
            C = phi * 1j * (r_m - q_m) * T_m + (kappa * theta / xs**2) * ((kappa - rho * xs * phi * 1j - d) * T_m - 2 * np.log((1 - g * et) / (1 - g + 1e-15)))
            
            kb = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            jump = lamb * T_m * (np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2) - 1 - 1j * phi * kb)
            return np.exp(C + D * v0 + jump)

        cf1, cf2 = get_cf(u - 1j), get_cf(u)
        int1 = np.real((cf1 * np.exp(-1j * u * np.log(K_m / S0) - (r_m - q_m) * T_m)) / (1j * u))
        int2 = np.real((cf2 * np.exp(-1j * u * np.log(K_m / S0))) / (1j * u))

        P1_v = 0.5 + (1/np.pi) * np.sum(0.5 * (int1[:-1, :] + int1[1:, :]) * du, axis=0)
        P2_v = 0.5 + (1/np.pi) * np.sum(0.5 * (int2[:-1, :] + int2[1:, :]) * du, axis=0)

        is_otm = (K_m > F_m)
        P1, P2 = np.where(is_otm, P1_v, 1.0 - P1_v), np.where(is_otm, P2_v, 1.0 - P2_v)
        
        pr = np.where(is_otm, S0 * np.exp(-q_m * T_m) * P1 - K_m * np.exp(-r_m * T_m) * P2,
                             K_m * np.exp(-r_m * T_m) * P2 - S0 * np.exp(-q_m * T_m) * P1)
        
        res = pr.flatten()
        is_put = (np.array([t.upper() for t in types]) == "PUT")
        if np.any(is_put != (~is_otm.flatten())):
            adj = S0 * np.exp(-q * T) - K * np.exp(-r * T)
            res = np.where(is_put, np.where(~is_otm.flatten(), res, res - adj),
                                   np.where(is_otm.flatten(), res, res + adj))
        
        return np.nan_to_num(np.maximum(res, 0.0))

class BatesAnalyticalPricerFast:
    @staticmethod
    def price_vectorized(S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        K, T, r, q, types = map(np.atleast_1d, [K, T, r, q, types])
        N_nodes, u_max = 1000, 20000 #250, 500 #126, 300.0
        nodes, weights = leg.leggauss(N_nodes)
        u, du = 0.5 * u_max * (nodes + 1), 0.5 * u_max * weights
        u = np.maximum(u, 1e-12)
        
        t_groups = {}
        for i, val in enumerate(T):
            t_groups.setdefault(val, []).append(i)
            
        prices, xs, kb = np.zeros(len(K)), np.maximum(xi, 1e-6), np.exp(mu_j + 0.5 * sigma_j**2) - 1
        
        def eval_cf(phi, tv, rv, qv):
            d = np.sqrt((kappa - rho * xs * phi * 1j)**2 + xs**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xs * phi * 1j - d) / (kappa - rho * xs * phi * 1j + d)
            et = np.exp(-d * tv)
            D = ((kappa - rho * xs * phi * 1j - d) / xs**2) * ((1 - et) / (1 - g * et))
            C = phi * 1j * (rv - qv) * tv + (kappa * theta / xs**2) * ((kappa - rho * xs * phi * 1j - d) * tv - 2 * np.log((1 - g * et)/(1 - g + 1e-15)))
            jump = lamb * tv * (np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2) - 1 - 1j * phi * kb)
            return np.exp(C + D * v0 + jump)

        for tv, idxs in t_groups.items():
            rv, qv = r[idxs[0]], q[idxs[0]]
            cf1, cf2 = eval_cf(u - 1j, tv, rv, qv), eval_cf(u, tv, rv, qv)
            ks, log_ks = K[idxs], np.log(K[idxs] / S0)[None, :]
            
            p1 = 0.5 + (1/np.pi) * np.sum(np.real((cf1[:, None] * np.exp(-1j * u[:, None] * log_ks - (rv - qv) * tv)) / (1j * u[:, None])) * du[:, None], axis=0)
            p2 = 0.5 + (1/np.pi) * np.sum(np.real((cf2[:, None] * np.exp(-1j * u[:, None] * log_ks)) / (1j * u[:, None])) * du[:, None], axis=0)
            
            is_otm = ks > (S0 * np.exp((rv - qv) * tv))
            P1, P2 = np.where(is_otm, p1, 1.0 - p1), np.where(is_otm, p2, 1.0 - p2)
            
            potm = np.where(is_otm, S0 * np.exp(-qv * tv) * P1 - ks * np.exp(-rv * tv) * P2,
                                   ks * np.exp(-rv * tv) * P2 - S0 * np.exp(-qv * tv) * P1)
            
            is_put = np.array([t.upper() == "PUT" for t in types[idxs]])
            adj = S0 * np.exp(-qv * tv) - ks * np.exp(-rv * tv)
            prices[idxs] = np.where(is_put, np.where(~is_otm, potm, potm - adj), np.where(is_otm, potm, potm + adj))

        return np.nan_to_num(np.maximum(prices, 0.0))