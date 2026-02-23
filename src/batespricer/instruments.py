from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from dataclasses import dataclass

class OptionType(Enum):
    CALL = 1
    PUT = -1

class Option(ABC):
    def __init__(self, K: float, T: float, option_type: OptionType):
        self.K, self.T, self.option_type = K, T, option_type

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        pass

class EuropeanOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T, phi = prices[:, -1], self.option_type.value
        return np.maximum(phi * (S_T - self.K), 0)
    
class AsianOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        avg_S, phi = np.mean(prices[:, 1:], axis=1), self.option_type.value
        return np.maximum(phi * (avg_S - self.K), 0)
    
class BarrierType(Enum):
    DOWN_AND_OUT = 1
    DOWN_AND_IN = 2
    UP_AND_OUT = 3
    UP_AND_IN = 4

class BarrierOption(Option):
    def __init__(self, K: float, T: float, barrier: float, barrier_type: BarrierType, option_type: OptionType):
        super().__init__(K, T, option_type)
        self.barrier, self.barrier_type = barrier, barrier_type

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T, phi = prices[:, -1], self.option_type.value
        payoff = np.maximum(phi * (S_T - self.K), 0)
        
        p_min, p_max = np.min(prices, axis=1), np.max(prices, axis=1)
        
        if self.barrier_type == BarrierType.DOWN_AND_OUT:
            mask = p_min > self.barrier
        elif self.barrier_type == BarrierType.DOWN_AND_IN:
            mask = p_min <= self.barrier
        elif self.barrier_type == BarrierType.UP_AND_OUT:
            mask = p_max < self.barrier
        elif self.barrier_type == BarrierType.UP_AND_IN:
            mask = p_max >= self.barrier
        else:
            return payoff
            
        return payoff * mask
    
@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: OptionType = OptionType.CALL