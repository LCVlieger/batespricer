# `batespricer`
Option pricer and calibrator for the Bates stochastic volatility jump-diffusion model. Calibrates to real-time options data, prices vanillas and path-dependent exotics, and computes Greeks. See the [report](batespricer.pdf) for the results and methodology.
## Pricing and calibration
Two pricing methods are implemented. The semi-analytical approach uses Fourier inversion of the Albrecher (2007) characteristic function. For this, two numerical approaches are implemented: midpoint direct integration and an accelerated version using Gauss-Legendre quadrature with maturity-based CF caching. The Monte Carlo approach uses full truncation (Lord et al., 2010) and quadratic exponential (Andersen, 2008) discretization. In the calibration the spread-weighted squared residuals are minimized, using the `L-BFGS-B` or `SLSQP` solvers. We calibrate to 300 OTM options per asset. The accelerated semi-analytical calibration converges in 10 seconds, with a price RMSE of 4.84 bps for the S&P 500 and 6.28 bps for Apple. Results under the calibrated Bates model (T = 1, K = 1.05 · S₀, B = 0.8 · S₀) are given by:
| | | SPX ($6,923) | | | | AAPL ($278) | | |
|---|---|---|---|---|---|---|---|---|
| **Product** | **Price** | **Δ** | **Γ** | **V**_var | **Price** | **Δ** | **Γ** | **V**_var |
| European call | $406.02 | 0.633 | 0.0004 | 3141.7 | $28.03 | 0.607 | 0.0060 | 97.50 |
| Down-Out call | $393.46 | 0.636 | 0.0004 | 2759.4 | $26.38 | 0.630 | 0.0046 | 70.93 |
| Down-In call | $12.56 | −0.003 | 0.0000 |  382.3 | $1.65 | −0.023 | 0.0014 | 26.57 |
| Asian call | $134.46 | 0.466 | 0.0009 | 2842.5 | $12.32 | 0.490 | 0.0112 | 85.89 |
## Package structure
```
src/batespricer/
├── analytics.py        # Semi-analytical pricers 
├── calibration.py      # BatesCalibrator, BatesCalibratorFast,
│                         BatesCalibratorMC, BatesCalibratorMCFast
├── data.py             # FRED yield curves (NSS-OLS), implied dividends, yfinance
├── instruments.py      # European, Asian, Barrier (Down-and-Out, In)
├── market.py           # MarketEnvironment dataclass
└── models/
    ├── mc_kernels.py   # Numba path generators
    ├── mc_pricer.py    # Monte Carlo pricer, Greeks
    └── process.py      # Black-Scholes, Heston, Bates process definitions
```
## Usage
```bash
git clone https://github.com/LCVlieger/batespricer
cd batespricer
pip install -e .
```
**Calibrate to market data for the four implementations:**
```bash
jupyter lab examples/1_calibration.ipynb
```
**Price exotics under calibrated parameters:**
```bash
jupyter lab examples/2_exotic_pricing.ipynb

```
## Tests
```bash
pytest tests/ -v
```
Validates naive and accelerated analytical pricers, verifies put-call parity, and tests instrument payoffs.
## References
- **Bates, D.S. (1996).** Jumps and stochastic volatility. *Review of Financial Studies*. 
- **Albrecher, H. et al. (2007).** The little Heston trap. *Wilmott Magazine*. 
- **Kilin, F. (2007).** Accelerating the calibration of stochastic volatility models. *CPQF Working Paper*. 
- **Andersen, L. (2008).** Efficient simulation of the Heston stochastic volatility model. *J. Computational Finance*.  
- **Lord, R. et al. (2010).** A comparison of biased simulation schemes. *Quantitative Finance*. 
- **Gatheral, J. (2006).** *The Volatility Surface*. Wiley. 
