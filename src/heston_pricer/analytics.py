import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

class BlackScholesPricer:
    @staticmethod
    def price_european_call(S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def price_asian_arithmetic_approximation(S0, K, T, r, sigma):
        """
        Turnbull-Wakeman (1991) approximation for arithmetic Asian options (ref: Hull CH26.13, p.626).
        Validates Monte Carlo priced asian options under GBM paths. 
        Matches the first two moments of the arithmetic average to a Lognormal distribution. 
        """
        # 1. Moments of the arithmetic average 
        if abs(r) < 1e-6:
            M1 = S0
            M2 = S0**2 * (2 * np.exp(sigma**2 * T) - 1)
        else:
            M1 = (np.exp(r * T) - 1) / (r * T) * S0
            
            term1 = 2 * np.exp((2 * r + sigma**2) * T) / ((r + sigma**2) * (2 * r + sigma**2) * T**2)
            term2 = (2 / (r * T**2)) * (
                1 / (2 * r + sigma**2) - 
                np.exp(r * T) / (r + sigma**2)
            )
            M2 = S0**2 * (term1 + term2)

        # 2. Match lognormal
        # v_eff^2 = ln(E[A^2] / E[A]^2)
        if M2 <= M1**2: return 0.0
            
        # 3. Pricing 
        sigma_eff = np.sqrt(np.log(M2 / M1**2) / T)
        d1 = (np.log(M1 / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)
        
        return np.exp(-r * T) * (M1 * norm.cdf(d1) - K * norm.cdf(d2))
    
class HestonAnalyticalPricer:
    """
    Semi-analytic Heston European Pricing.
    Implements Albrecher (2007) stable forms to avoid the 'Heston Trap'.
    """
    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        # 1. Parameter constraints for numerical stability
        # Prevent zero-division or instability in extreme skew regimes
        xi = max(xi, 1e-4)
        
        def heston_char_func(u):
            # Albrecher (2007) Stable Form
            # d: root with positive real part (numpy.sqrt ensures this for complex inputs)
            d = np.sqrt((rho * xi * u * 1j - kappa)**2 + xi**2 * (u * 1j + u**2))
            
            # g (auxiliary variable)
            # The choice of d (Re(d)>0) ensures |g| <= 1 usually, but we must handle the log carefully
            g = (kappa - rho * xi * u * 1j - d) / (kappa - rho * xi * u * 1j + d)
            
            # Characteristic Function Exponents
            C = (1/xi**2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)) * \
                (kappa - rho * xi * u * 1j - d)
                
            # STABLE D CALCULATION:
            # Split the log to avoid the branch cut of the quotient np.log(A/B)
            # D = (kappa*theta/xi^2) * [ ... - 2*ln((1-g*e^-dT)/(1-g)) ] <-- Unstable
            val_num = 1 - g * np.exp(-d * T)
            val_denom = 1 - g
            
            # Separating logs avoids the spiral crossing the branch cut of the quotient
            D = (kappa * theta / xi**2) * \
                ((kappa - rho * xi * u * 1j - d) * T - 2 * (np.log(val_num) - np.log(val_denom)))
            
            # Dividend/Risk-free adjustment (Drift)
            drift_term = 1j * u * np.log(S0 * np.exp((r - q) * T))
            
            return np.exp(C * v0 + D + drift_term)

        # 2. Dynamic Integration Limit based on Maturity
        # Short maturities require larger u to decay.
        # Approx rule: u_max ~ 100 / (xi * T)
        # We cap it to avoid performance kills, but 950 is too low for T < 0.1
        inv_T = 1.0 / max(T, 1e-4)
        upper_bound = max(1000.0, 150.0 * inv_T) * 2
        upper_bound = min(upper_bound, 50000.0) # Safety cap
        
        # 3. Integrands with Singularity Handling (u=0)
        def integrand_p1(u):
            # P1 uses phi(u - i)
            num = np.exp(-1j * u * np.log(K)) * heston_char_func(u - 1j)
            denom = 1j * u * S0 * np.exp((r - q) * T)
            return np.real(num / denom)
            
        def integrand_p2(u):
            # P2 uses phi(u)
            num = np.exp(-1j * u * np.log(K)) * heston_char_func(u)
            denom = 1j * u
            return np.real(num / denom)
            
        # 4. Integration
        # ERROR FIX: Start at 1e-8, not 0, to avoid ZeroDivisionError/NaN at the limit
        # 'limit' is the max number of subdivisions. 500 is usually plenty if the function is continuous.
        try:
            P1 = 0.5 + (1/np.pi) * integrate.quad(integrand_p1, 1e-8, upper_bound, limit=3000)[0]
            P2 = 0.5 + (1/np.pi) * integrate.quad(integrand_p2, 1e-8, upper_bound, limit=3000)[0]
        except Exception:
            # Fallback if integration fails (e.g. extreme parameters)
            return max(S0 - K, 0.0) # Intrinsic

        # 5. Final Assembly
        price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        return max(price, 0.0)

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        """
        Prices a European Put using Put-Call Parity.
        P = C - S*exp(-qT) + K*exp(-rT)
        """
        call_price = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
        return call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)