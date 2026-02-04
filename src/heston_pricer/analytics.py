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
    Extends the Heston model with Merton Log-Normal Jumps, maintaining Albrecher et al. (2007)
    numerical stability.
    """
    
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0, 
                                       lamb, mu_j, sigma_j):
        """
        Calculates European Call option prices using vectorized Fourier integration.
        
        Parameters:
        -----------
        ... [Standard Heston Parameters] ...
        lamb : float
            Jump intensity (lambda). Frequency of jumps per year.
        mu_j : float
            Mean of log-jump size.
        sigma_j : float
            Standard deviation of log-jump size.
            
        Returns:
        --------
        np.ndarray
            Array of computed option prices.
        """
        # High density grid matching your optimized implementation
        N_grid, u_max = 1000, 200.0
        du = u_max / N_grid
        # Shape: (N_grid, 1)
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 
        
        # Ensure inputs are at least 1D arrays
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        
        # Broadcast inputs to shape (1, N_options) to vectorize against integration grid
        T_mat = T[np.newaxis, :]
        r_mat = r[np.newaxis, :]
        q_mat = q[np.newaxis, :]
        K_mat = K[np.newaxis, :]

        def get_cf(phi):
            # Stability: Ensure xi is not zero
            xi_s = np.maximum(xi, 1e-6)
            
            # --- 1. Heston Component (Albrecher 2007) ---
            d = np.sqrt((rho * xi_s * phi * 1j - kappa)**2 + xi_s**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            
            exp_neg_dT = np.exp(-d * T_mat)
            
            C = (1/xi_s**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT)) * (kappa - rho * xi_s * phi * 1j - d)
            
            # Log subtraction for branch-cut stability
            D = (kappa * theta / xi_s**2) * ((kappa - rho * xi_s * phi * 1j - d) * T_mat - 
                2 * (np.log(1 - g * exp_neg_dT) - np.log(1 - g + 1e-15)))
            
            # --- 2. Bates Jump Component ---
            # Martingale Compensator (Mean expected jump size)
            # k_bar = E[e^J] - 1
            k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            
            # Jump Characteristic Function part:
            # Term: lambda * T * ( E[e^{i*phi*J}] - 1 - i*phi*k_bar )
            # The (- i*phi*k_bar) is the drift correction to keep risk-neutrality
            
            # E[e^{i*phi*J}] for Normal J ~ N(mu_j, sigma_j^2)
            # formula: exp(i*phi*mu - 0.5*phi^2*sigma^2)
            # Note: phi is complex here, so phi**2 handles the sign correctly
            e_i_phi_J = np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2)
            
            jump_part = lamb * T_mat * (e_i_phi_J - 1 - 1j * phi * k_bar)

            # --- 3. Final Assembly ---
            # Drift uses the actual S0 price (r-q is handled here)
            drift = 1j * phi * np.log(S0 * np.exp((r_mat - q_mat) * T_mat))
            
            return np.exp(C * v0 + D + drift + jump_part)

        # Calculate Characteristic Functions for P1 and P2
        cf_p1 = get_cf(u - 1j)
        cf_p2 = get_cf(u)
        
        # Integration logic using raw K and S0
        int_p1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p1) / (1j * u * S0 * np.exp((r_mat - q_mat) * T_mat)))
        int_p2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p2) / (1j * u))
        
        # Gauss-Lobatto style summation
        P1 = 0.5 + (1/np.pi) * np.sum(int_p1 * du, axis=0)
        P2 = 0.5 + (1/np.pi) * np.sum(int_p2 * du, axis=0)
        
        # Final Price: S * e^{-qT} * P1 - K * e^{-rT} * P2
        price = S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2
        
        return np.nan_to_num(np.maximum(price.flatten(), 0.0), nan=0.0)
    
    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        return float(BatesAnalyticalPricer.price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0)[0])

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        call = BatesAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)