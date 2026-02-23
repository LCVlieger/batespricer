import pytest
import numpy as np
from batespricer.market import MarketEnvironment
from batespricer.instruments import EuropeanOption, AsianOption, OptionType
from batespricer.models.process import BlackScholesProcess
from batespricer.models.mc_pricer import MonteCarloPricer
from batespricer.analytics import BlackScholesPricer

@pytest.fixture
def default_market():
    return MarketEnvironment(S0=100, r=0.05, q=0.0, sigma=0.2)

def test_european_call_convergence(default_market):
    T, K = 1.0, 100
    option = EuropeanOption(K=K, T=T, option_type=OptionType.CALL)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    result = pricer.price(option, n_paths=200_000, n_steps=100)
    
    bs_p = BlackScholesPricer.price_european_call(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    assert abs(result.price - bs_p) < 0.05

def test_asian_call_approximation(default_market):
    T, K = 1.0, 100
    option = AsianOption(K=K, T=T, option_type=OptionType.CALL)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    result = pricer.price(option, n_paths=200_000, n_steps=100)
    
    tw_p = BlackScholesPricer.price_asian_arithmetic_approximation(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    assert abs(result.price - tw_p) < 0.20

def test_put_call_parity(default_market):
    T, K = 1.0, 100
    call = EuropeanOption(K, T, OptionType.CALL)
    put = EuropeanOption(K, T, OptionType.PUT)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    c_p = pricer.price(call, n_paths=100_000, n_steps=100).price
    p_p = pricer.price(put, n_paths=100_000, n_steps=100).price
    
    rhs = default_market.S0 - K * np.exp(-default_market.r * T)
    
    assert abs((c_p - p_p) - rhs) < 0.15