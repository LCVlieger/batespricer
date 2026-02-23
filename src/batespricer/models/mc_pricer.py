import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, Tuple
from ..instruments import Option
from .process import StochasticProcess
from ..calibration import implied_volatility

@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval_95: tuple[float, float]

class MonteCarloPricer:
    def __init__(self, process: StochasticProcess):
        self.process = process

    def price(self, option: Option, n_paths: int = 10000, n_steps: int = 100, **kwargs) -> PricingResult:
        paths = self.process.generate_paths(option.T, n_paths, n_steps, **kwargs)
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
            Z_CRN[0] = rng.standard_normal((n_steps, n_paths)) # Asset
            Z_CRN[1] = rng.standard_normal((n_steps, n_paths)) # Vol
            Z_CRN[2] = rng.random((n_steps, n_paths))          # Jump Trigger
            Z_CRN[3] = rng.standard_normal((n_steps, n_paths)) # Jump Size
        else:
            Z_CRN = rng.standard_normal((n_chan, n_steps, n_paths))
        
        # Base Case
        res_curr = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Delta & Gamma bumps
        self.process.market = replace(mkt, S0 = S0 + eps_s)
        res_up = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        self.process.market = replace(mkt, S0 = S0 - eps_s)
        res_down = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Vega bump
        self.process.market = replace(mkt, v0 = v0 + eps_v, S0 = S0)
        res_vega = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Restore and Calculate
        self.process.market = mkt
        
        delta = (res_up.price - res_down.price) / (2 * eps_s)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (eps_s ** 2)
        vega = (res_vega.price - res_curr.price) / eps_v
        
        return {
            "price": res_curr.price,
            "delta": delta,
            "gamma": gamma,
            "vega_v0": vega
        }