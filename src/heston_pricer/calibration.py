import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Union
from scipy.interpolate import interp1d
from collections import defaultdict

# Internal imports - Ensure these exist in your project structure
# Assuming these imports work in your env based on previous context
from .analytics import HestonAnalyticalPricer
from .market import MarketEnvironment
from .models.process import HestonProcess

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
        # Handle single rate case (flat curve)
        if len(tenors) == 1:
            self.curve = lambda t: rates[0]
        else:
            self.curve = interp1d(
                tenors, rates, 
                kind='linear', 
                fill_value="extrapolate" 
            )

    def get_rate(self, T: float) -> float:
        if T < 1e-5: 
            # Avoid extrapolation errors at T=0
            return float(self.rates[0]) if self.tenors else 0.0
        return float(self.curve(T))

    def to_dict(self):
        return {"tenors": self.tenors, "rates": self.rates}

def implied_volatility(price: float, S: float, K: float, T: float, r: float, q: float, option_type: str = "CALL") -> float:
    if price <= 0: return 0.0
    
    # Intrinsic check
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

# --- ANALYTICAL CALIBRATOR ---
class HestonCalibrator:
    def __init__(self, S0: float, r_curve: SimpleYieldCurve, q_curve: SimpleYieldCurve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve # Now a Curve object

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        x0 = init_guess if init_guess else [2.0, 0.05, 0.3, -0.7, 0.04]
        # (lower_bound, upper_bound)
        bounds =  [(0.1, 15.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]

        def objective(params):
            kappa, theta, xi, rho, v0 = params
            
            penalty = 0.0
            # Feller violation penalty
            if 2 * kappa * theta < xi**2:
                penalty += 0.0 * ((xi**2 - 2 * kappa * theta)**2)

            total_error = 0.0
            for opt in options:
                # Term Structure Lookups
                r_T = self.r_curve.get_rate(opt.maturity)
                q_T = self.q_curve.get_rate(opt.maturity)
                
                # 1. Analytical Price using specific r_T and q_T
                if opt.option_type == "PUT":
                    model_p = HestonAnalyticalPricer.price_european_put(
                        self.S0, opt.strike, opt.maturity, r_T, q_T, kappa, theta, xi, rho, v0
                    )
                else:
                    model_p = HestonAnalyticalPricer.price_european_call(
                        self.S0, opt.strike, opt.maturity, r_T, q_T, kappa, theta, xi, rho, v0
                    )
                
                # 2. Moneyness Weighting
                moneyness = np.log(opt.strike / self.S0)
                wing_weight = 1.0 + 5.0 * (moneyness**2)
                
                # 3. Relative Error
                relative_error = (model_p - opt.market_price) / (opt.market_price + 1e-5)
                total_error += wing_weight * (relative_error**2)

            return total_error + penalty

        def callback(xk):
             print(f"   [Analytical] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)

        result = minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            callback=callback,
            tol=1e-7, options={'ftol': 1e-7, 'eps': 1e-7, 'maxiter': 100}
        )
        
        # Final Stats calculation
        kappa, theta, xi, rho, v0 = result.x
        sse_iv, count = 0.0, 0
        for opt in options:
            r_T = self.r_curve.get_rate(opt.maturity)
            q_T = self.q_curve.get_rate(opt.maturity)
            
            if opt.option_type == "PUT":
                model_price = HestonAnalyticalPricer.price_european_put(
                    self.S0, opt.strike, opt.maturity, r_T, q_T, kappa, theta, xi, rho, v0
                )
            else:
                model_price = HestonAnalyticalPricer.price_european_call(
                    self.S0, opt.strike, opt.maturity, r_T, q_T, kappa, theta, xi, rho, v0
                )
            
            iv_mkt = implied_volatility(opt.market_price, self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
            iv_model = implied_volatility(model_price, self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
            
            if iv_mkt > 0 and iv_model > 0:
                sse_iv += (iv_model - iv_mkt) ** 2
                count += 1
        
        return {
            "kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0,
            "success": result.success, "fun": result.fun, 
            "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
        }

# --- MONTE CARLO CALIBRATOR ---
class HestonCalibratorMC:
    def __init__(self, S0: float, r_curve: SimpleYieldCurve, q_curve: SimpleYieldCurve, n_paths: int = 30000, n_steps: int = 100):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve # Now a Curve object
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.z_noise = None 
        
        # Batching containers
        self.maturity_batches = defaultdict(list)
        self.max_T = 0.0
        self.dt = 0.0

    def _precompute_batches(self, options: List[MarketOption]):
        """Organizes options by maturity to handle term structures."""
        self.maturity_batches.clear()
        if not options: return
        self.max_T = max(opt.maturity for opt in options)
        self.dt = self.max_T / self.n_steps
        
        # 1. Group options
        for opt in options:
            self.maturity_batches[opt.maturity].append(opt)
            
        # 2. Pre-generate ONE giant noise block (Brownian Bridge consistency)
        if self.z_noise is None:
            np.random.seed(42) 
            self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params: List[float]) -> Dict[float, List[float]]:
        """Returns map: {maturity: [price_opt1, price_opt2...]}"""
        kappa, theta, xi, rho, v0 = params
        
        results = {}
        
        # Loop over unique maturities
        for T_target, opts in self.maturity_batches.items():
            
            # A. Get rate/div for this specific maturity
            r_T = self.r_curve.get_rate(T_target)
            q_T = self.q_curve.get_rate(T_target)
            
            # B. Determine steps needed for this T
            steps_needed = int(round(T_target / self.dt))
            if steps_needed < 1: steps_needed = 1
            if steps_needed > self.n_steps: steps_needed = self.n_steps
            
            # C. Setup Environment & Process
            env = MarketEnvironment(self.S0, r_T, q_T)
            process = HestonProcess(env)
            process.market.kappa = kappa
            process.market.theta = theta
            process.market.xi = xi
            process.market.rho = rho
            process.market.v0 = v0
            
            # D. Simulation
            # Slice noise to match time-steps
            noise_slice = self.z_noise[:, :steps_needed, :]
            paths = process.generate_paths(
                T=T_target, n_paths=self.n_paths, n_steps=steps_needed, noise=noise_slice
            )
            S_final = paths[:, -1]
            
            # E. Pricing
            prices = []
            for opt in opts:
                if opt.option_type == "PUT":
                    payoff = np.maximum(opt.strike - S_final, 0.0)
                else:
                    payoff = np.maximum(S_final - opt.strike, 0.0)
                
                # Discount using the specific rate r_T
                price = np.mean(payoff) * np.exp(-r_T * T_target)
                prices.append(price)
                
            results[T_target] = prices
            
        return results

    def objective(self, params):
        kappa, theta, xi, rho, v0 = params
        
        penalty = 0.0
        if 2 * kappa * theta < xi**2:
            penalty += 0.0 * ((xi**2 - 2 * kappa * theta)**2)

        # Get prices for all maturities
        model_prices_map = self.get_prices(params)
        
        total_error = 0.0
        
        # Match Analytical Weighting Logic
        for T, opts in self.maturity_batches.items():
            m_prices = model_prices_map[T]
            
            for i, opt in enumerate(opts):
                model_p = m_prices[i]
                
                moneyness = np.log(opt.strike / self.S0)
                wing_weight = 1.0 + 5.0 * (moneyness**2)
                
                relative_error = (model_p - opt.market_price) / (opt.market_price + 1e-5)
                total_error += wing_weight * (relative_error**2)

        return total_error + penalty

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        self._precompute_batches(options)
        x0 = init_guess if init_guess else [2.0, 0.05, 0.3, -0.7, 0.04]
        bounds = [(0.1, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]
        
        def callback(xk):
             print(f"   [MonteCarlo] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)
        
        result = minimize(
            self.objective, x0, method='L-BFGS-B', bounds=bounds, 
            callback=callback, 
            tol=1e-5, options={'ftol': 1e-5, 'eps': 1e-5, 'maxiter': 50}
        )

        # Final IV Statistics
        final_map = self.get_prices(result.x)
        sse_iv, count = 0.0, 0
        
        for T, opts in self.maturity_batches.items():
            r_T = self.r_curve.get_rate(T)
            q_T = self.q_curve.get_rate(T) # Look up Q for this T
            m_prices = final_map[T]
            for i, opt in enumerate(opts):
                iv_mkt = implied_volatility(opt.market_price, self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                iv_model = implied_volatility(m_prices[i], self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                
                if iv_mkt > 0 and iv_model > 0:
                    sse_iv += (iv_model - iv_mkt) ** 2
                    count += 1
        
        return {
            "kappa": result.x[0], "theta": result.x[1], "xi": result.x[2],
            "rho": result.x[3], "v0": result.x[4], 
            "success": result.success, 
            "fun": result.fun, 
            "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
        }