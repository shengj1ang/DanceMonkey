# Soft-Blending Decoding Pipeline Evolution

| Version | Decoding Method | RMSE | Time To Run | Score |
|--------|----------------|------|-------------|-------|
| v1 | Baseline implementation | 8.0271 | 3.44 | 7.56839 |
| v2 | Direction buffering changed: use decodedHandPos to detect new trajectory instead of trialId | 8.0043 | 3.454 | 7.54927 |
| v3 | Features use cumulative + recent window instead of full history; regression predicts residuals relative to mean trajectory | 10.0308 | 3.1374 | 9.34146 |
| v4 (2026-03-10 submission) | Direction: 8-direction soft blending; very low compute (O(1) dot products); excellent trajectory smoothness with temporal memory; strong noise robustness (EMA + Ridge) | 7.825 | 3.4192 | 7.38442 |
| v5 (2026-03-17 submission) | Further tuning; stronger PCA compression; smoother direction blending; position drift nearly removed; stronger start-point constraint | 7.5008 | 3.1728 | 7.068 |

---

## Notes

- Previous SVM-based approach reached a performance ceiling, so the pipeline was redesigned.
- New direction: **PCA + LDA + kNN hybrid direction modeling**.

---

## Design Comparison

| Feature | Original Version | Optimized Version |
|--------|----------------|------------------|
| Direction Estimation | Hard switch (single best direction) | 8-direction weighted soft blending |
| Computational Cost | High (frequent high-dimensional transforms) | Very low (O(1) vector dot products) |
| Trajectory Smoothness | Poor, prone to abrupt changes | Excellent, with temporal smoothing memory |
| Noise Robustness | Weak (sensitive to outliers) | Strong (EMA + Ridge regularization) |
