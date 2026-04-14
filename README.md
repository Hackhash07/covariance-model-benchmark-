# covariance-model-benchmark-

# Covariance Estimation Benchmark for Portfolio Optimization

## Overview
This project compares different covariance estimation techniques for portfolio optimization:

- MLE (Sample Covariance)
- Constant Correlation Model (CCM)
- Multi-Group Constant Correlation Model (MGCCM)

The goal is to evaluate how well estimated covariance matrices generalize to future data.

---

## Methodology
1. Load stock return data
2. Split into train (70%) and test (30%)
3. Estimate covariance using different models
4. Construct Minimum Variance Portfolio (MVP)
5. Evaluate using:
   - Frobenius Norm (estimation error)
   - Realized Volatility (out-of-sample risk)

---

## Key Insight
Models that perform well on training data may fail on real market data, highlighting instability in covariance estimation.

---

## How to Run
```bash
pip install -r requirements.txt
python main.py
