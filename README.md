# `batespricer`

Option pricer and calibrator under the Bates stochastic volatility model. Calibrates to live market data, prices vanillas and path-dependent exotics, and computes Greeks.

The Bates model extends Heston by adding Merton log-normal jumps to the asset price process. This captures both the volatility smile at shorter maturities and the skew at longer maturities. For a detailed description of the methodology and results, see the [report](batespricer.pdf).

## Pricing and calibration

Two pricing approaches are implemented. A semi-analytical direct integration method uses the Attari (2004) formulation with Gauss-Legendre quadrature and maturity-based caching. A Monte Carlo simulation uses full truncation (Lord et al., 2010) and quadratic exponential (Andersen, 2008) discretization schemes. Calibration is formulated as spread-weighted nonlinear least squares, solved with L-BFGS-B and SLSQP. The accelerated semi-analytical calibration converges in under 10 seconds. Calibrated to 300 OTM options per asset, the semi-analytical approach achieves a price RMSE of 4.84 bps for the S&P 500 and 6.28 bps for Apple.

The calibrated parameters are applied to exotic options. Results under the Bates model for the S&P 500 and Apple (T = 1, K = 1.05 · S₀, B = 0.8 · S₀):

| | | SPX ($6,923) | | | AAPL ($278) | | |
|---|---|---|---|---|---|---|---|
| **Product** | **Price** | **Δ** | **Γ** | **Price** | **Δ** | **Γ** | **V**_var |
| European call | $406.02 | 0.633 | 0.0004 | $28.03 | 0.607 | 0.0060 | 97.50 |
| Down&Out call | $393.46 | 0.636 | 0.0004 | $26.38 | 0.630 | 0.0046 | 70.93 |
| Down&In call | $12.56 | −0.003 | 0.0000 | $1.65 | −0.023 | 0.0014 | 26.57 |
| Asian call | $134.46 | 0.466 | 0.0009 | $12.32 | 0.490 | 0.0112 | 85.89 |

## Package structure

```
src/batespricer/
├── analytics.py        # Semi-analytical pricers (naive midpoint + cached Gauss-Legendre)
├── calibration.py      # BatesCalibrator, BatesCalibratorFast,
│                         BatesCalibratorMC, BatesCalibratorMCFast
├── data.py             # FRED yield curves (NSS-OLS), implied dividends, yfinance
├── instruments.py      # European, Asian, Barrier (Down-and-Out/In)
├── market.py           # MarketEnvironment dataclass
└── models/
    ├── mc_kernels.py   # Numba-JIT path generators (full truncation + QE)
    ├── mc_pricer.py    # Monte Carlo pricer with CRN-based finite difference Greeks
    └── process.py      # Black-Scholes, Heston, Bates process definitions
```

## Usage

```bash
git clone https://github.com/LCVlieger/batespricer
cd batespricer
pip install -e .
```

**Calibrate to market data:**
```bash
python examples/1a_market_calibration_Ana_naive.py
python examples/1b_market_calibration_Ana_accelerated.py
python examples/1c_market_calibration_MC_naive.py
python examples/1d_market_calibration_MC_accelerated.py
```

**Price exotics under calibrated parameters:**
```bash
python examples/2_exotic_pricing.py
```

## Tests

```bash
pytest tests/ -v
```

Validates European call convergence to Black-Scholes, Asian call convergence to Turnbull-Wakeman, and put-call parity.

## References

- Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. *J. Political Economy*, 81(3), 637–654.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *J. Financial Economics*, 3(1-2), 125–144.
- Gil-Pelaez, J. (1951). Note on the inversion theorem. *Biometrika*, 38(3-4), 481–482.
- Nelson, C.R. and Siegel, A.F. (1987). Parsimonious modeling of yield curves. *J. Business*, 60(4), 473–489.
- Turnbull, S.M. and Wakeman, L.M. (1991). A quick algorithm for pricing European average options. *J. Financial and Quantitative Analysis*, 26(3), 377–389.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343.
- Svensson, L.E.O. (1994). Estimating and interpreting forward interest rates: Sweden 1992–1994. *NBER Working Paper* 4871.
- Bates, D.S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options. *Review of Financial Studies*, 9(1), 69–107.
- Carr, P. and Madan, D.B. (1999). Option valuation using the fast Fourier transform. *J. Computational Finance*, 2(4), 61–73.
- Attari, M. (2004). Option pricing using Fourier transforms: a numerically efficient simplification. *SSRN Working Paper*.
- Albrecher, H. et al. (2007). The little Heston trap. *Wilmott Magazine*, Jan, 83–92.
- Andersen, L. (2008). Efficient simulation of the Heston stochastic volatility model. *J. Computational Finance*, 11(3), 1–22.
- Lord, R. et al. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance*, 10(2), 177–194.