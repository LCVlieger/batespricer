import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict
from scipy.interpolate import interp1d
from collections import defaultdict
import warnings

# --- DATA STRUCTURES ---
@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 

class SimpleYieldCurve:
    def __init__(self, tenors: List[float], rates: List[float]):
        self.tenors = tenors
        self.rates = rates
        if len(tenors) == 1:
            self.curve = lambda t: rates[0]
        else:
            self.curve = interp1d(tenors, rates, kind='linear', fill_value="extrapolate")

    def get_rate(self, T: float) -> float:
        if T < 1e-5: return float(self.rates[0]) if self.tenors else 0.0
        return float(self.curve(T))

    def to_dict(self):
        return {"tenors": self.tenors, "rates": self.rates}

@dataclass
class MarketEnvironment:
    S0: float
    r: float
    q: float
    kappa: float = 0.0
    theta: float = 0.0
    xi: float = 0.0
    rho: float = 0.0
    v0: float = 0.0

# --- ANALYTICAL CALIBRATOR (Notebook Replication Mode) ---
class HestonCalibrator:
    def __init__(self, S0: float, r_curve: SimpleYieldCurve, q_curve: SimpleYieldCurve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        """
        Calibrates the 6-parameter Heston model using the EXACT bounds 
        and formulas from the reference notebook to prevent overflow.
        """
        strikes = np.array([opt.strike for opt in options])
        maturities = np.array([opt.maturity for opt in options])
        market_prices = np.array([opt.market_price for opt in options])
        
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        # --- CRITICAL FIX: EXACT NOTEBOOK BOUNDS ---
        # The notebook restricts sigma to [0.01, 1.0]. 
        # Allowing it > 1.0 causes the "overflow" crash you saw.
        bounds = [
            (1e-3, 0.1),  # v0
            (1e-3, 5.0),  # kappa (Notebook max: 5)
            (1e-3, 0.1),  # theta
            (1e-2, 1.0),  # sigma (Notebook max: 1.0 - PREVENTS EXPLOSION)
            (-1.0, 0.0),  # rho
            (-1.0, 1.0)   # lambd
        ]
        
        # Initial guess from notebook
        x0 = init_guess if init_guess else [0.1, 3.0, 0.05, 0.3, -0.8, 0.03]

        def objective(params):
            v0, kappa, theta, sigma, rho, lambd = params
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore') # Suppress overflow warnings during optimization
                    
                    model_prices = _heston_price_notebook_style(
                        self.S0, strikes, maturities, r_vec, q_vec,
                        v0, kappa, theta, sigma, rho, lambd
                    )
                    
                    # Handle NaNs if the optimizer pushes too hard
                    if np.any(np.isnan(model_prices)) or np.any(np.isinf(model_prices)):
                        return 1e9
                    
                    return np.mean((model_prices - market_prices)**2)
            except:
                return 1e9

        def callback(xk):
             print(f"   [Analytical] v0={xk[0]:.4f}, k={xk[1]:.2f}, theta={xk[2]:.4f}, sigma={xk[3]:.4f}, rho={xk[4]:.2f}, lambd={xk[5]:.4f}", flush=True)

        print("Starting 6-Parameter Minimization (Notebook Replication Mode)...")
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, callback=callback, tol=1e-4)

        return {
            "v0": float(result.x[0]), "kappa": float(result.x[1]), "theta": float(result.x[2]),
            "sigma": float(result.x[3]), "rho": float(result.x[4]), "lambd": float(result.x[5]),
            "fun": float(result.fun), "success": bool(result.success)
        }

# --- HELPERS: EXACT NOTEBOOK FORMULAS ---
def _heston_price_notebook_style(S0, K, tau, r, q, v0, kappa, theta, sigma, rho, lambd):
    # Grid: N=10000 matches notebook precision
    N, umax = 10000, 100
    dphi = umax / N
    phi = dphi * (2 * np.arange(1, N) + 1) / 2
    phi = phi[:, np.newaxis] 

    # Integrand logic
    # We pass 'q' effectively as an adjustment to 'r' in drift terms if needed,
    # or handle it in the final pricing formula like the user's notebook structure.
    
    # Calculate characteristic functions
    # Note: To replicate the notebook exactly, we use its specific algebra
    
    args = (S0, tau, r, q, v0, kappa, theta, sigma, rho, lambd)
    
    # C = 1/2(S0*e^-qT - K*e^-rT) + 1/pi * integral
    term1 = 0.5 * (S0 * np.exp(-q * tau) - K * np.exp(-r * tau))
    
    # Calculate Integrands
    # The notebook evaluates: exp(-i*phi*k) * phi(phi) / (i*phi) 
    # But it splits it into two terms. We use the combined form for stability.
    
    # Replicating the Notebook's "Integrand" function logic exactly:
    # numerator = exp(r*tau) * heston_charfunc(phi-1j) - K * heston_charfunc(phi)
    # denominator = 1j * phi * K^(1j*phi)
    
    # Evaluate Phi(phi - i)
    numer_1 = np.exp(r * tau) * _char_func_standard(phi - 1j, *args)
    # Evaluate Phi(phi)
    numer_2 = K * _char_func_standard(phi, *args)
    
    numerator = numer_1 - numer_2
    denominator = 1j * phi * (K**(1j * phi))
    
    integral_sum = np.real(np.sum(dphi * numerator / denominator, axis=0))
    
    return term1 + integral_sum / np.pi

def _char_func_standard(phi, S0, tau, r, q, v0, kappa, theta, sigma, rho, lambd):
    """Standard Heston Characteristic Function (Notebook Version)."""
    a = kappa * theta
    b = kappa + lambd
    rspi = rho * sigma * phi * 1j

    d = np.sqrt((rspi - b)**2 + (phi * 1j + phi**2) * sigma**2)
    g = (b - rspi + d) / (b - rspi - d)

    # Notebook uses (r) here. We include (r-q) to handle European index properly.
    exp1 = np.exp((r - q) * phi * 1j * tau)
    
    # Careful with power operations on complex bases
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
    
    exp2 = np.exp(a * tau * (b - rspi + d) / sigma**2 + 
                  v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2)
    
    return exp1 * term2 * exp2

# --- UTILS (Unchanged) ---
def implied_volatility(price, S, K, T, r, q, option_type="CALL"):
    if price <= 0: return 0.0
    if option_type == "PUT":
        intrinsic = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0)
    else:
        intrinsic = max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if price < intrinsic: return 0.0

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "PUT":
             val = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))
        else:
             val = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return val - price
    
    try:
        return brentq(bs_price, 0.001, 5.0)
    except:
        return 0.0

# --- MONTE CARLO COMPONENTS (Unchanged) ---
class MarketEnvironment:
    def __init__(self, S0, r, q, kappa=0, theta=0, xi=0, rho=0, v0=0):
        self.S0, self.r, self.q = S0, r, q
        self.kappa, self.theta, self.xi, self.rho, self.v0 = kappa, theta, xi, rho, v0

class HestonProcess:
    def __init__(self, market):
        self.market = market

    def generate_paths(self, T, n_paths, n_steps, noise):
        dt = T / n_steps
        S = np.full(n_paths, self.market.S0)
        v = np.full(n_paths, self.market.v0)
        S_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = S

        for t in range(n_steps):
            z1 = noise[0, t, :]
            z2 = self.market.rho * z1 + np.sqrt(1 - self.market.rho**2) * noise[1, t, :]
            v = np.maximum(v, 0)
            v = v + self.market.kappa * (self.market.theta - v) * dt + self.market.xi * np.sqrt(v * dt) * z2
            v = np.maximum(v, 0)
            S = S * np.exp((self.market.r - self.market.q - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
            S_paths[:, t+1] = S
        return S_paths

class HestonCalibratorMC:
    def __init__(self, S0, r_curve, q_curve, n_paths=30000, n_steps=100):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve
        self.n_paths, self.n_steps = n_paths, n_steps
        self.z_noise, self.maturity_batches = None, defaultdict(list)
        self.max_T, self.dt = 0.0, 0.0

    def _precompute_batches(self, options):
        self.maturity_batches.clear()
        if not options: return
        self.max_T = max(opt.maturity for opt in options)
        self.dt = self.max_T / self.n_steps
        for opt in options: self.maturity_batches[opt.maturity].append(opt)
        if self.z_noise is None:
            np.random.seed(42) 
            self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params):
        kappa, theta, xi, rho, v0 = params
        results = {}
        for T_target, opts in self.maturity_batches.items():
            r_T, q_T = self.r_curve.get_rate(T_target), self.q_curve.get_rate(T_target)
            steps = max(1, min(self.n_steps, int(round(T_target / self.dt))))
            env = MarketEnvironment(self.S0, r_T, q_T, kappa, theta, xi, rho, v0)
            process = HestonProcess(env)
            paths = process.generate_paths(T_target, self.n_paths, steps, self.z_noise[:, :steps, :])
            S_final = paths[:, -1]
            prices = [np.mean(np.maximum(opt.strike - S_final, 0) if opt.option_type == "PUT" else np.maximum(S_final - opt.strike, 0)) * np.exp(-r_T * T_target) for opt in opts]
            results[T_target] = prices
        return results

    def objective(self, params):
        model_prices_map = self.get_prices(params)
        total_error = 0.0
        for T, opts in self.maturity_batches.items():
            m_prices = model_prices_map[T]
            for i, opt in enumerate(opts):
                moneyness = np.log(opt.strike / self.S0)
                weight = 1.0 + 5.0 * (moneyness**2)
                total_error += weight * ((m_prices[i] - opt.market_price) / (opt.market_price + 1e-8))**2
        return total_error

    def calibrate(self, options, init_guess=None):
        self._precompute_batches(options)
        x0 = init_guess if init_guess else [2.0, 0.05, 0.3, -0.7, 0.04]
        bounds = [(0.1, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]
        def callback(xk):
             print(f"   [MonteCarlo] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)
        result = minimize(self.objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-5)
        final_map = self.get_prices(result.x)
        sse_iv, count = 0.0, 0
        for T, opts in self.maturity_batches.items():
            r_T, q_T, m_prices = self.r_curve.get_rate(T), self.q_curve.get_rate(T), final_map[T]
            for i, opt in enumerate(opts):
                iv_mkt = implied_volatility(opt.market_price, self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                iv_model = implied_volatility(m_prices[i], self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                if iv_mkt > 0 and iv_model > 0:
                    sse_iv += (iv_model - iv_mkt) ** 2
                    count += 1
        return {"kappa": result.x[0], "theta": result.x[1], "xi": result.x[2], "rho": result.x[3], "v0": result.x[4], "success": result.success, "fun": result.fun, "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0}