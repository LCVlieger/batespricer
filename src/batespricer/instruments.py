from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from dataclasses import dataclass

class OptionType(Enum):
    CALL = 1
    PUT = -1

class Option(ABC):
    """Base class for payoff definitions."""
    def __init__(self, K: float, T: float, option_type: OptionType):
        self.K, self.T, self.option_type = K, T, option_type

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        pass

class EuropeanOption(Option):
    """Vanilla European payoff at expiry."""
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T, phi = prices[:, -1], self.option_type.value
        return np.maximum(phi * (S_T - self.K), 0)
    
class AsianOption(Option):
    """Arithmetic average-price Asian payoff."""
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        avg_S, phi = np.mean(prices[:, 1:], axis=1), self.option_type.value
        return np.maximum(phi * (avg_S - self.K), 0)
    
class BarrierType(Enum):
    DOWN_AND_OUT = 1
    DOWN_AND_IN = 2
    UP_AND_OUT = 3
    UP_AND_IN = 4
class BarrierOption(Option):
    """Down-and-Out / Down-and-In barrier with optional smooth payoff."""
    def __init__(self, K: float, T: float, barrier: float, barrier_type: BarrierType, option_type: OptionType):
        super().__init__(K, T, option_type)
        self.barrier, self.barrier_type = barrier, barrier_type

    def payoff(self, paths: np.ndarray, epsilon: float = None) -> np.ndarray:
        if self.option_type == OptionType.CALL:
            vanilla_payoff = np.maximum(paths[:, -1] - self.K, 0)
        else:
            vanilla_payoff = np.maximum(self.K - paths[:, -1], 0)
            
        if self.barrier_type == BarrierType.DOWN_AND_OUT:
            min_S = np.min(paths, axis=1)
            if epsilon is None:
                survived = np.where(min_S > self.barrier, 1.0, 0.0)                
            else:
                buffer_size = 3.0 * epsilon
                x = np.clip((min_S - self.barrier) / buffer_size, 0.0, 1.0)
                survived = (3 * x**2) - (2 * x**3)
                
            return vanilla_payoff * survived
            
        elif self.barrier_type == BarrierType.DOWN_AND_IN:
            min_S = np.min(paths, axis=1)
            if epsilon is None:
                survived = np.where(min_S <= self.barrier, 1.0, 0.0)
            else:
                buffer_size = 3.0 * epsilon
                x = np.clip((min_S - self.barrier) / buffer_size, 0.0, 1.0)
                survived = 1.0 - ((3 * x**2) - (2 * x**3))
                
            return vanilla_payoff * survived
            
        else:
            raise NotImplementedError("Only Down-and-Out and Down-and-In are supported")
