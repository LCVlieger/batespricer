from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..market import MarketEnvironment
from .mc_kernels import (
    generate_paths_kernel, 
    generate_heston_paths, 
    generate_heston_paths_crn, 
    generate_bates_paths, 
    generate_bates_paths_crn
)

class StochasticProcess(ABC):
    """Base class for path-generating stochastic processes."""
    def __init__(self, market: MarketEnvironment):
        self.market = market

    @abstractmethod
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        pass

class BlackScholesProcess(StochasticProcess):
    """GBM with constant volatility."""
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        m = self.market
        return generate_paths_kernel(m.S0, m.r, m.q, m.sigma, T, n_paths, n_steps)

class HestonProcess(StochasticProcess):
    """Heston (1993) stochastic variance with full truncation."""
    def generate_paths(self, T: float, n_paths: int, n_steps: int, noise=None) -> np.ndarray:
        m = self.market
        args = (m.S0, m.r, m.q, m.v0, m.kappa, m.theta, m.xi, m.rho, T, n_paths, n_steps)
        
        if noise is not None:
            return generate_heston_paths_crn(*args, noise)
        return generate_heston_paths(*args)
    
class BatesProcess(StochasticProcess):
    """Heston + Merton log-normal jumps (Bates 1996)."""
    @property
    def noise_channels(self) -> int:
        return 4 

    def generate_paths(self, T: float, n_paths: int, n_steps: int, noise=None) -> np.ndarray:
        m = self.market
        la = getattr(m, 'lamb', 0.0)
        mj = getattr(m, 'mu_j', 0.0)
        sj = getattr(m, 'sigma_j', 0.0)
        
        args = (
            m.S0, m.r, m.q, m.v0, m.kappa, m.theta, m.xi, m.rho, 
            la, mj, sj, T, n_paths, n_steps
        )
        
        if noise is not None:
            return generate_bates_paths_crn(*args, noise)
        return generate_bates_paths(*args)