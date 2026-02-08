import numpy as np
from numba import jit

# --- KERNELS ---

@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, q: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Classical GBM Black-Scholes kernel. Includes Antithetic Sampling.
    """
    dt = T / n_steps
    half_paths = n_paths // 2
    
    # Variance reduction: Antithetic variates [Z, -Z]
    Z = np.concatenate((np.random.standard_normal((half_paths, n_steps)), 
                        -np.random.standard_normal((half_paths, n_steps))), axis=0)
    
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    
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
    """
    Heston Model simulation using a truncated Euler discretization.
    FIXED: Ensures Asset update uses v_t (not v_{t+1}) to align with Ito calculus.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0, dtype=np.float64)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        
        # CORRECTED: Capture v_t before updating to v_{t+1}
        v_t = np.maximum(curr_v, 0.0)
        
        # Update Asset using v_t
        curr_s *= np.exp((r - q - 0.5 * v_t) * dt + np.sqrt(v_t) * sqrt_dt * Z1)

        # Update Variance using v_t (Euler)
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        
        prices[:, j + 1] = curr_s
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, noise_matrix):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = noise_matrix[0, j]
        Z2 = noise_matrix[1, j]
        Zv = c1 * Z1 + c2 * Z2
        
        # --- FIX: Capture v_t BEFORE updating ---
        v_t = np.maximum(curr_v, 0.0) 
        
        # Update Asset using v_t (Standard Euler)
        # drift = (r - q - 0.5 * v_t) * dt
        # diffusion = sqrt(v_t) * sqrt(dt) * Z1
        curr_s *= np.exp((r - q - 0.5 * v_t) * dt + np.sqrt(v_t) * sqrt_dt * Z1)
        
        # Update Variance using v_t
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        
        prices[:, j + 1] = curr_s
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_paths(S0, r, q, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, T, n_paths, n_steps):
    """
    Bates Model (Heston + Merton Jump Diffusion) simulation.
    Uses Poisson-Gaussian jumps.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    # Expected relative jump size (compensator for drift)
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_correction = lamb * k
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        
        # --- JUMP COMPONENT ---
        # Number of jumps in this step (Poisson)
        # Note: For small lambda*dt, this is mostly 0 or 1.
        N_jumps = np.random.poisson(lamb * dt, n_paths)
        
        # Total Log-Jump size: Sum of N normals -> N(N*mu, N*sigma^2)
        # We simulate this by scaling a single standard normal by sqrt(N)
        Z_jump = np.random.standard_normal(n_paths)
        
        # Avoid sqrt(0) warnings/errors implicitly by multiplication (or masking if strictly needed)
        # N_jumps is integer, so we cast to float for sqrt
        jump_magnitude = N_jumps * mu_j + np.sqrt(N_jumps) * sigma_j * Z_jump
        
        # --- HESTON UPDATE ---
        v_t = np.maximum(curr_v, 0.0)
        
        # Asset Update:
        # drift = r - q - 0.5*v_t - lambda*k
        drift_term = (r - q - 0.5 * v_t - drift_correction) * dt
        diff_term = np.sqrt(v_t) * sqrt_dt * Z1
        
        curr_s *= np.exp(drift_term + diff_term + jump_magnitude)
        
        # Variance Update (Standard Heston)
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        
        prices[:, j + 1] = curr_s
        
    return prices
@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, T, n_paths, n_steps, noise_matrix):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    # Calculate Compensator Internally
    k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_correction = lamb * k_bar
    jump_prob = lamb * dt

    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = noise_matrix[0, j, :n_paths] 
        Z2 = noise_matrix[1, j, :n_paths]
        U_jump = noise_matrix[2, j, :n_paths] 
        Z_size = noise_matrix[3, j, :n_paths]
        
        Zv = c1 * Z1 + c2 * Z2
        v_t = np.maximum(curr_v, 0.0)
        
        jump_mag = np.where(U_jump < jump_prob, mu_j + sigma_j * Z_size, 0.0)
        
        # FIX: Include r, q, and drift_correction
        drift_term = (r - q - 0.5 * v_t - drift_correction) * dt 
        diff_term = np.sqrt(v_t) * sqrt_dt * Z1
        
        curr_s *= np.exp(drift_term + diff_term + jump_mag)
        
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        prices[:, j + 1] = curr_s
        
    return prices