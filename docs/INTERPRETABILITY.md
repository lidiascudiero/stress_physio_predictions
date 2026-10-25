# Interpretability

The interpretation workflow completed **1,485 fold analyses across
99 saved experiment entries**, including complete-model reruns.
These are not 99 independent model designs.

Explanations are computed on held-out participants using **Monte Carlo
Shapley estimates with antithetic feature-order sampling** and training-only
reference data. The implementation does not require SHAP or Numba and
is **not exact TreeSHAP**.

Selected findings include:

- **Wrist binary classifiers:** pulse rate and PRV SDNN were the leading
  individual features, with the BVP/PRV family showing the largest grouped
  contribution.
- **Original wrist Random Forest:** temperature mean had the largest
  mean absolute contribution, followed by EDA variability and movement.
- **Reduced early-fusion Random Forest:** chest activity level and ECG
  beat count were prominent, alongside wrist movement and EDA.
- **Four-class early-fusion Random Forest:** respiratory rate was the
  highest-ranked individual feature.
- **Late fusion:** grouped chest and wrist contributions were similar
  in magnitude, although this does not establish equal sensor utility.

Beeswarm, waterfall, PDP and first-order ALE plots support both global
and individual inspection. Rank stability varies by configuration:
the original wrist Random Forest had mean pairwise rank Spearman
**0.734**, compared with **0.660** for late fusion.

All algorithms are explained on the class-probability scale, but different
tasks and reference distributions remain distinct attribution problems.
Monte Carlo standard errors are saved; additivity checks alone do not
establish convergence.

These findings describe **model reliance, not physiological causes or
validated biomarkers**. Correlated inputs and task-related movement
can complicate interpretation.
