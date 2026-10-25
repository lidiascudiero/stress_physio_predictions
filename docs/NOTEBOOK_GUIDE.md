# Notebook guide

| Notebook | Focus |
|---|---|
| `00` | Environment and dependencies |
| `01` | Dataset loading and audit |
| `02` | Signal QC, extraction, and ACC lag diagnostics |
| `03` / `03b` | Feature tables, distributions, group comparisons, and correlations |
| `04` | Original primary LOSO classification |
| `05` | Secondary tasks and exploratory extensions |
| `06` | Calibration, uncertainty, and rejection analysis |
| `07` | Report, limitations, and self-report processing |
| `08` | Model/self-report associations |
| `09` / `09b` | Correlation sensitivity, tree importance, and saved reduced models |
| `10` / `10b` |Focused interpretation of six headline configurations across 90 LOSO folds, including Shapley summaries, beeswarm, waterfall, PDP and ALE plots |
| `10_all_loso_experiments_interpretability_no_numba.ipynb` | Optional comprehensive interpretation of all supported saved LOSO experiments |
| `10_training_block_permutation_loso.ipynb` | Conditional training-only block-randomization diagnostic |
| `11` | Metric audit, visual benchmark, and explicit ranking rules |

The selected-model interpretability notebook provides a compact reporting
view of primary, secondary and exploratory configurations. These are not
six statistically established winners. It loads saved models without
retraining and estimates probability-scale Shapley values without
importing SHAP or Numba.

The selected and comprehensive interpretability notebooks are alternative
scopes; running both is not required.

