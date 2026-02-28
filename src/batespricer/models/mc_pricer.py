import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, Tuple
from ..instruments import Option
from .process import StochasticProcess
from ..analytics import implied_volatility

@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval_95: tuple[float, float]

class MonteCarloPricer:
    def __init__(self, process: StochasticProcess):
        self.process = process
    def price(self, option: Option, n_paths: int = 10000, n_steps: int = 100, **kwargs) -> PricingResult:
        epsilon = kwargs.pop('epsilon', None)
        
        paths = self.process.generate_paths(option.T, n_paths, n_steps, **kwargs)
        try:
            payoffs = option.payoff(paths, epsilon=epsilon)
        except TypeError:
            payoffs = option.payoff(paths)
    
        discount = np.exp(-self.process.market.r * option.T)
        disc_payoffs = payoffs * discount
    
        mu = np.mean(disc_payoffs)
        se = np.std(disc_payoffs, ddof=1) / np.sqrt(n_paths)
        
        return PricingResult(
            price=mu,
            std_error=se,
            conf_interval_95=(mu - 1.96 * se, mu + 1.96 * se)
        )
    def compute_greeks(self, option: Option, n_paths: int = 10000, n_steps: int = 252, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        mkt = self.process.market
        S0, v0 = mkt.S0, mkt.v0
        eps_s, eps_v = S0 * bump_ratio, 0.001 
        
        n_chan = getattr(self.process, 'noise_channels', 2)
        rng = np.random.default_rng(seed)
        
        if n_chan == 4:
            Z_CRN = np.zeros((4, n_steps, n_paths))
            Z_CRN[0] = rng.standard_normal((n_steps, n_paths)) 
            Z_CRN[1] = rng.standard_normal((n_steps, n_paths)) 
            Z_CRN[2] = rng.random((n_steps, n_paths))          
            Z_CRN[3] = rng.standard_normal((n_steps, n_paths)) 
        else:
            Z_CRN = rng.standard_normal((n_chan, n_steps, n_paths))
        res_curr_exact = self.price(option, n_paths, n_steps, noise=Z_CRN)
        res_curr_smoothed = self.price(option, n_paths, n_steps, noise=Z_CRN, epsilon=eps_s)
        
        # Delta & Gamma bumps
        self.process.market = replace(mkt, S0 = S0 + eps_s)
        res_up = self.price(option, n_paths, n_steps, noise=Z_CRN, epsilon=eps_s)
        
        self.process.market = replace(mkt, S0 = S0 - eps_s)
        res_down = self.price(option, n_paths, n_steps, noise=Z_CRN, epsilon=eps_s)
        
        # Vega bumps 
        v_down = max(v0 - eps_v, 1e-6) 
        actual_eps_v_down = v0 - v_down

        self.process.market = replace(mkt, v0 = v0 + eps_v, S0 = S0)
        res_vega_up = self.price(option, n_paths, n_steps, noise=Z_CRN, epsilon=eps_s)

        self.process.market = replace(mkt, v0 = v_down, S0 = S0)
        res_vega_down = self.price(option, n_paths, n_steps, noise=Z_CRN, epsilon=eps_s)

        # Restore Market State
        self.process.market = mkt
        
        # Calculate Greeks using the SMOOTHED base case
        delta = (res_up.price - res_down.price) / (2 * eps_s)
        gamma = (res_up.price - 2 * res_curr_smoothed.price + res_down.price) / (eps_s ** 2)
        vega = (res_vega_up.price - res_vega_down.price) / (eps_v + actual_eps_v_down)
        
        return {
            "price": res_curr_exact.price,    # Return the exact, unsmoothed price
            "delta": delta,
            "gamma": gamma,
            "vega_v0": vega
        }