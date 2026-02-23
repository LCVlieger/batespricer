from dataclasses import dataclass

@dataclass
class MarketEnvironment:
    S0: float          
    r: float           
    q: float = 0.0     
    sigma: float = 0.2 
     
    # Heston/Bates Parameters
    v0: float = 0.04
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 0.1
    rho: float = -0.7
    
    # Jump Component
    lamb: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0