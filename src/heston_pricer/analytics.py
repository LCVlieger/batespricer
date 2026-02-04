from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq


def implied_volatility(price, S, K, T, r, q, option_type="CALL"): 
    if price <= 0: return 0.0
    intrinsic = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0) if option_type == "PUT" else max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if price < intrinsic: return 0.0
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        val = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)) if option_type == "PUT" else (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return val - price
    try: return brentq(bs_price, 0.001, 5.0)
    except: return 0.0
class BatesAnalyticalPricer:
    """
    High-performance analytical pricer for European options using the Bates (1996) model.
    Extends the Heston model with Merton Log-Normal Jumps.
    """

    @staticmethod
    def price_vectorized(S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j):
        """
        Prices a mixed vector of Calls and Puts using Put-Call Parity.
        K, T, r, q, and types must all be arrays of the same length.
        """
        # 1. Calculate everything as Calls first (Vectorized)
        calls = BatesAnalyticalPricer.price_european_call_vectorized(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
        )
        
        # 2. Identify Puts
        is_put = (types == "PUT")
        
        if np.any(is_put):
            # Apply Put-Call Parity: P = C - S*e^-qT + K*e^-rT
            puts = calls - S0 * np.exp(-q * T) + K * np.exp(-r * T)
            return np.where(is_put, puts, calls)
        
        return calls

    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0, 
                                       lamb, mu_j, sigma_j):
        """
        Vectorized Fourier integration engine for European Calls.
        """
        N_grid, u_max = 1000, 200.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 
        
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        
        T_mat = T[np.newaxis, :]
        r_mat = r[np.newaxis, :]
        q_mat = q[np.newaxis, :]
        K_mat = K[np.newaxis, :]

        def get_cf(phi):
            xi_s = np.maximum(xi, 1e-6)
            
            # --- Heston Component ---
            d = np.sqrt((rho * xi_s * phi * 1j - kappa)**2 + xi_s**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            exp_neg_dT = np.exp(-d * T_mat)
            
            C = (1/xi_s**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT)) * (kappa - rho * xi_s * phi * 1j - d)
            D = (kappa * theta / xi_s**2) * ((kappa - rho * xi_s * phi * 1j - d) * T_mat - 
                2 * (np.log(1 - g * exp_neg_dT) - np.log(1 - g + 1e-15)))
            
            # --- Bates Jump Component ---
            k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            e_i_phi_J = np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2)
            jump_part = lamb * T_mat * (e_i_phi_J - 1 - 1j * phi * k_bar)

            drift = 1j * phi * np.log(S0 * np.exp((r_mat - q_mat) * T_mat))
            return np.exp(C * v0 + D + drift + jump_part)

        cf_p1 = get_cf(u - 1j)
        cf_p2 = get_cf(u)
        
        int_p1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p1) / (1j * u * S0 * np.exp((r_mat - q_mat) * T_mat)))
        int_p2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p2) / (1j * u))
        
        P1 = 0.5 + (1/np.pi) * np.sum(int_p1 * du, axis=0)
        P2 = 0.5 + (1/np.pi) * np.sum(int_p2 * du, axis=0)
        
        price = S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2
        return np.nan_to_num(np.maximum(price.flatten(), 0.0), nan=0.0)
    
    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j):
        return float(BatesAnalyticalPricer.price_european_call_vectorized(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)[0])

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j):
        call = BatesAnalyticalPricer.price_european_call(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)