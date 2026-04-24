# Decoding Methods Comparison (Hard Pipeline)

| Version | Decoding Method | RMSE | Time To Run | Score |
|--------|----------------|------|-------------|-------|
| v1 (2026-03-04 submission) | Template cosine (320ms) + per-direction Ridge LR (velocity) + integration | 21.8186 | 1.5657 | 19.79331 |
| v2 | Template cosine (direction) + per-direction Ridge LR (velocity) + Kalman filter | 23.1723 | 1.7553 | 21.0306 |
| v3 | KNN direction + Ridge (velocity) + Kalman | 23.1175 | 1.7115 | 20.9769 |
| v4 | Same as v1 but stores a re-classification time for decoding | 23.1723 | 1.6504 | 21.02011 |
| v5 | Template cosine (direction) + PCA + Ridge (velocity) + Kalman | 21.0919 | 1.8459 | 19.1673 |
| v6 | LSTM (direction @320ms) + per-direction Ridge (velocity) + Kalman | 84.2372 | 3.2692 | 76.1404 |
| v7 | SVM direction + Ridge velocity + Kalman | 22.2799 | 3.5872 | 20.41063 |
| v8 | SVM direction + Ridge velocity + learned 6D Kalman Filter with acceleration | 22.2278 | 2.0826 | 20.21328 |
| v9 | SVM direction + Ridge velocity + learned 6D Constant-Acceleration Kalman Filter (Early/Late Q,R) | 22.0896 | 2.6074 | 20.14138 |

---

**Baseline:** Template-based direction + velocity ridge regression + integration  
(Fast, but direction/velocity noise accumulates and causes drift)

**Direction stabilization + filtering:**  
Tried LDA / KNN / re-classification / PCA / LSTM for improving direction features.  
The most effective strategy was **more stable direction + trajectory-level filtering**.
