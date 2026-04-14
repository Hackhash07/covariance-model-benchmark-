# Covariance Estimation Benchmark for Portfolio Optimization

## Overview

This project benchmarks different covariance estimation techniques used in portfolio optimization:

* **MLE (Sample Covariance)**
* **Constant Correlation Model (CCM)**
* **Multi-Group Constant Correlation Model (MGCCM)**

The objective is to evaluate how well these models generalize to unseen (out-of-sample) market data.

---

## Methodology

1. Load historical stock price data and compute log returns
2. Split data into **training (70%)** and **testing (30%)**
3. Estimate covariance matrices using different models
4. Construct **Minimum Variance Portfolio (MVP)**
5. Evaluate performance using:

   * **Frobenius Norm** → estimation error
   * **Realized Volatility** → out-of-sample risk

---

## Results (Sample Output)

```
Model   F-Norm      Vol     Time
MLE     0.002802  0.010959  0.000058
CCM     0.002810  0.011155  0.003839
MGCCM   0.002798  0.010901  0.002678
```

---

## Key Insights

* **MGCCM achieves the lowest estimation error and realized volatility**, indicating better generalization
* **MLE is fastest**, but slightly less accurate
* **CCM underperforms due to oversimplified correlation assumptions**

This highlights the trade-off between **model complexity, accuracy, and computational cost**.

---

## Conclusion

MGCCM provides improved portfolio performance by incorporating **sector-level structure**, though at a higher computational cost compared to MLE.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Skills Demonstrated

* Quantitative Finance
* Covariance Modeling
* Portfolio Optimization
* Python for Financial Analysis
* Out-of-sample Validation

---

## Future Improvements

* Extend to larger asset universe
* Incorporate shrinkage estimators (Ledoit-Wolf)
* Evaluate Sharpe ratio and drawdowns
