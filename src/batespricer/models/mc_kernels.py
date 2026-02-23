import numpy as np
from numba import jit
import math

@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, q: float, sigma: float, T: float, n_paths: int, n_steps: int) -> np.ndarray:
    dt = T / n_steps
    hp = n_paths // 2
    
    Z = np.concatenate((np.random.standard_normal((hp, n_steps)), 
                        -np.random.standard_normal((hp, n_steps))), axis=0)
    
    drift, diff = (r - q - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    for i in range(n_paths):
        log_ret = 0.0
        for j in range(n_steps):
            log_ret += drift + diff * Z[i, j]
            prices[i, j + 1] = S0 * np.exp(log_ret)
            
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    dt, sqrt_dt = T / n_steps, np.sqrt(T / n_steps)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    cv, cs = np.full(n_paths, v0), np.full(n_paths, S0, dtype=np.float64)
    
    for j in range(n_steps):
        Z1, Z2 = np.random.standard_normal(n_paths), np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        vt = np.maximum(cv, 0.0)
        
        cs *= np.exp((r - q - 0.5 * vt) * dt + np.sqrt(vt) * sqrt_dt * Z1)
        cv += kappa * (theta - vt) * dt + xi * np.sqrt(vt) * sqrt_dt * Zv
        prices[:, j + 1] = cs
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, noise_matrix):
    dt, sqrt_dt = T / n_steps, np.sqrt(T / n_steps)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1)); prices[:, 0] = S0
    cv, cs = np.full(n_paths, v0), np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1, Z2 = noise_matrix[0, j], noise_matrix[1, j]
        vt = np.maximum(cv, 0.0) 
        cs *= np.exp((r - q - 0.5 * vt) * dt + np.sqrt(vt) * sqrt_dt * Z1)
        cv += kappa * (theta - vt) * dt + xi * np.sqrt(vt) * sqrt_dt * (c1 * Z1 + c2 * Z2)
        prices[:, j + 1] = cs
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_paths(S0, r, q, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, T, n_paths, n_steps):
    dt, sqrt_dt = T / n_steps, np.sqrt(T / n_steps)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    k_comp = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_corr = lamb * k_comp
    
    prices = np.zeros((n_paths, n_steps + 1)); prices[:, 0] = S0
    cv, cs = np.full(n_paths, v0), np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1, Z2 = np.random.standard_normal(n_paths), np.random.standard_normal(n_paths)
        nj = np.random.poisson(lamb * dt, n_paths)
        zj = np.random.standard_normal(n_paths)
        jump_m = nj * mu_j + np.sqrt(nj.astype(np.float64)) * sigma_j * zj
        
        vt = np.maximum(cv, 0.0)
        cs *= np.exp((r - q - 0.5 * vt - drift_corr) * dt + np.sqrt(vt) * sqrt_dt * Z1 + jump_m)
        cv += kappa * (theta - vt) * dt + xi * np.sqrt(vt) * sqrt_dt * (c1 * Z1 + c2 * Z2)
        prices[:, j + 1] = cs
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, T, n_paths, n_steps, noise_matrix):
    dt, sqrt_dt = T / n_steps, np.sqrt(T / n_steps)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_corr = lamb * k_bar
    
    prices = np.zeros((n_paths, n_steps + 1)); prices[:, 0] = S0
    cv, cs = np.full(n_paths, v0, dtype=np.float64), np.full(n_paths, S0, dtype=np.float64)
    lam_dt, en_lam = lamb * dt, np.exp(-lamb * dt)
    
    for j in range(n_steps):
        Z1, Z2, Uj, Zs = noise_matrix[0, j, :n_paths], noise_matrix[1, j, :n_paths], noise_matrix[2, j, :n_paths], noise_matrix[3, j, :n_paths]
        vt = np.maximum(cv, 0.0)
        cv += kappa * (theta - vt) * dt + xi * np.sqrt(vt) * sqrt_dt * (c1 * Z1 + c2 * Z2)
        db, diff = (r - q - 0.5 * vt - drift_corr) * dt, np.sqrt(vt) * sqrt_dt * Z1
        
        for i in range(n_paths):
            u, nj, pp = Uj[i], 0, en_lam
            sp = pp
            while u > sp and nj < 10:
                nj += 1
                pp *= lam_dt / nj
                sp += pp
            jm = nj * mu_j + np.sqrt(float(nj)) * sigma_j * Zs[i] if nj > 0 else 0.0
            cs[i] *= np.exp(db[i] + diff[i] + jm)
        prices[:, j + 1] = cs
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_qe_slices_crn(S0, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, dt, n_paths, n_steps, maturity_step_idxs, noise_matrix):
    n_mats = len(maturity_step_idxs)
    m_prices = np.zeros((n_paths, n_mats), dtype=np.float64)
    
    kb, exk = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0, math.exp(-kappa * dt)
    c1, c2 = xi**2 * exk / kappa * (1.0 - exk), theta * xi**2 * 0.5 / kappa * (1.0 - exk)**2
    r_inv_xi, s1r2 = rho / xi, math.sqrt(1.0 - rho**2)
    
    cv, cl_s = np.full(n_paths, v0, dtype=np.float64), np.full(n_paths, math.log(S0), dtype=np.float64)
    lam_dt, en_lam, next_mi = lamb * dt, math.exp(-lamb * dt), 0
    ts = maturity_step_idxs[next_mi]
    
    for j in range(n_steps):
        Zv, Zx, Zj, Uv, Uj = noise_matrix[0, j, :], noise_matrix[1, j, :], noise_matrix[2, j, :], noise_matrix[3, j, :], noise_matrix[4, j, :]
        for i in range(n_paths):
            vt = cv[i]
            m, s2 = theta + (vt - theta) * exk, vt * c1 + c2
            psi = s2 / (m**2)
            if psi <= 1.5:
                b2 = 2.0 / psi - 1.0 + math.sqrt(2.0 / psi * (2.0 / psi - 1.0))
                vn = (m / (1.0 + b2)) * (math.sqrt(b2) + Zv[i])**2
            else:
                p = (psi - 1.0) / (psi + 1.0)
                vn = math.log((1.0 - p) / (1.0 - Uv[i])) / ((1.0 - p) / m) if Uv[i] > p else 0.0
            cv[i] = vn
            
            vi = 0.5 * (vt + vn) * dt 
            ls_dr = -lamb * kb * dt - 0.5 * vi + r_inv_xi * (vn - vt - kappa * theta * dt + kappa * vi)
            ls_df = s1r2 * math.sqrt(vi) * Zx[i]
            
            u, nj, pp = Uj[i], 0, en_lam
            sp = pp
            while u > sp and nj < 10:
                nj += 1; pp *= lam_dt / nj; sp += pp
            jm = nj * mu_j + math.sqrt(float(nj)) * sigma_j * Zj[i] if nj > 0 else 0.0
            cl_s[i] += ls_dr + ls_df + jm

        while next_mi < n_mats and (j + 1) == ts:
            for i in range(n_paths): m_prices[i, next_mi] = math.exp(cl_s[i])
            next_mi += 1
            if next_mi < n_mats: ts = maturity_step_idxs[next_mi]
    return m_prices