"""
Covariance Estimation Benchmark

Author: Saketh

Compares MLE, CCM, and MGCCM covariance models for portfolio optimization.

Workflow:
1. Load stock return data
2. Estimate covariance matrices
3. Construct Minimum Variance Portfolio (MVP)
4. Evaluate using out-of-sample metrics
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import norm
import os
import time


# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data(stock_files, base_path):
    returns_list = []

    for stock, filename in stock_files.items():
        path = os.path.join(base_path, filename)

        if not os.path.exists(path):
            print(f"Warning: {filename} not found, skipping.")
            continue

        df = pd.read_csv(path)

        # Use 'Close' if available, else fallback
        col = 'Close' if 'Close' in df.columns else df.columns[1]

        returns = np.log(df[col] / df[col].shift(1)).dropna()
        returns_list.append(returns)

    data = pd.concat(returns_list, axis=1).dropna().values

    # Train-test split (70-30)
    split_idx = int(len(data) * 0.7)
    return data[:split_idx], data[split_idx:]


# ==========================================
# 2. COVARIANCE MODELS
# ==========================================
def get_mle_cov(train_data):
    """Sample covariance (MLE)"""
    return (train_data.T @ train_data) / len(train_data), None


def get_ccm_cov(train_data):
    """Constant Correlation Model (CCM)"""
    n = train_data.shape[1]

    def ccm_ll(rho, returns):
        if rho <= -1/(n-1) or rho >= 1:
            return 1e10

        std = np.std(returns, axis=0)
        norm_r = returns / std

        term = (n-1)*np.log(1-rho) + np.log(1-rho+n*rho)

        quad = (
            np.sum(norm_r**2)
            - (rho/(1+(n-1)*rho)) * np.sum(np.sum(norm_r, axis=1)**2)
        ) / (1-rho)

        return 0.5 * (term + quad/len(returns))

    res = minimize(ccm_ll, x0=[0.2], args=(train_data,),
                   bounds=[(-0.24, 0.99)])

    std = np.std(train_data, axis=0)
    R = np.full((n, n), res.x[0])
    np.fill_diagonal(R, 1.0)

    cov = np.diag(std) @ R @ np.diag(std)
    return cov, res.x[0]


def get_mgccm_cov_fast(train_data, sector_indices):
    """Multi-Group CCM (sector-based correlation)"""
    n = train_data.shape[1]
    num_sectors = len(sector_indices)

    std = np.std(train_data, axis=0, ddof=0)
    C_sample = np.corrcoef(train_data, rowvar=False)

    def mgccm_objective(p):
        intra_rhos, rho_inter = p[:-1], p[-1]

        R = np.full((n, n), rho_inter)

        for i, s_idx in enumerate(sector_indices):
            R[np.ix_(s_idx, s_idx)] = intra_rhos[i]

        np.fill_diagonal(R, 1.0)

        try:
            L = np.linalg.cholesky(R)
            logdet = 2 * np.sum(np.log(np.diag(L)))
            trace_term = np.trace(np.linalg.solve(R, C_sample))
            return logdet + trace_term
        except np.linalg.LinAlgError:
            return 1e10

    res = minimize(
        mgccm_objective,
        x0=[0.4]*num_sectors + [0.2],
        bounds=[(0.01, 0.99)]*(num_sectors+1),
        method='L-BFGS-B',
        options={'ftol': 1e-5}
    )

    # Reconstruct covariance
    opt = res.x
    R_final = np.full((n, n), opt[-1])

    for i, s_idx in enumerate(sector_indices):
        R_final[np.ix_(s_idx, s_idx)] = opt[i]

    np.fill_diagonal(R_final, 1.0)

    cov = np.diag(std) @ R_final @ np.diag(std)
    return cov, opt


# ==========================================
# 3. PORTFOLIO OPTIMIZATION
# ==========================================
def calculate_mvp_weights(cov_matrix):
    """Minimum Variance Portfolio weights"""
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    return (inv_cov @ ones) / (ones.T @ inv_cov @ ones)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    # Stock files (place inside ./data/)
    stock_config = {
        'Reliance': 'reliance2.csv',
        'Powergrid': 'powergrid1.csv',
        'NTPC': 'ntpc1.csv',
        'HDFC': 'hdfc1.csv',
        'ICICI': 'icici1.csv'
    }

    # Sector grouping (indices)
    my_sectors = [[0, 1, 2], [3, 4]]

    # IMPORTANT: relative path for GitHub
    data_path = "./data/"

    # Load data
    train, test = load_data(stock_config, data_path)

    # True future covariance
    future_cov = (test.T @ test) / len(test)

    results = []

    models = [
        ('MLE', lambda d: get_mle_cov(d)),
        ('CCM', lambda d: get_ccm_cov(d)),
        ('MGCCM', lambda d: get_mgccm_cov_fast(d, my_sectors))
    ]

    for name, method in models:
        t0 = time.perf_counter()

        cov, _ = method(train)
        weights = calculate_mvp_weights(cov)

        t1 = time.perf_counter()

        f_norm = norm(cov - future_cov, ord='fro')
        realized_vol = np.sqrt(weights.T @ future_cov @ weights)

        results.append({
            'Model': name,
            'F-Norm': f_norm,
            'Vol': realized_vol,
            'Time': t1 - t0
        })

    print("\n--- N-STOCK BENCHMARK RESULTS ---")
    print(pd.DataFrame(results).to_string(index=False))
