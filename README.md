 Wearable Stress & Affect: From Physiological Signals to Interpretable Predictions

This repository implements a Machine Learning workflow for **subject-independent classification of laboratory stress and affect**, using wrist and chest recordings from the WESAD dataset.

The project connects physiological feature engineering, nested participant-level validation, multimodal fusion, uncertainty analysis, and model interpretation. The central question goes beyond which algorithm scores highest: **what information survives across people, how much does each sensor setup contribute, and where do the predictions become unreliable?**

This is an independent analysis of WESAD, not an official implementation or an exact reproduction of its original benchmark. Dataset collection and the experimental protocol belong to the original authors. [Original paper](https://doi.org/10.1145/3242969.3242985) · [Dataset](https://doi.org/10.24432/C57K5T)

---

## The “Neurocognitive Integrity” Angle

> **Author’s Note:** My background in **Cognitive Psychology and Cognitive Neuroscience** shaped how I approached this project. I treated physiological signals, experimental conditions, and subjective experience as related but distinct sources of information. A participant exposed to a stress task is not automatically experiencing a uniform level of stress, and a confident algorithm is not automatically a reliable one. This perspective motivated participant-independent evaluation, separate self-report analyses, and an explicit examination of individual differences. My aim was to connect computational predictions with psychological interpretation while keeping the limits of each measurement visible. These are **peripheral physiological recordings—not direct measurements of neural activity**.

---

## Project Workflow

- **Signal Audit & Quality Control:** Loaded the extracted recordings, checked participant coverage, signal availability, missing values, and window-level quality. Added ACC cross-correlation as a diagnostic for possible temporal offsets, without automatically realigning recordings.
- **Feature Engineering & Exploration:** Extracted cardiac, pulse, electrodermal, respiratory, temperature, movement, and muscle features. Explored distributions, condition differences, participant variability, outliers, and feature redundancy.
- **Subject-Independent Benchmarking:** Compared majority classification, logistic regression, Random Forest, XGBoost, LightGBM, and MLP using LOSO evaluation and inner cross-validation grouped by participant.
- **Multimodal Analysis:** Compared wrist, chest, early fusion, late fusion, and simulated missing-modality scenarios. Primary sensor comparisons used the same common valid windows.
- **Robustness & Reliability:** Evaluated post-hoc correlation filtering, participant-level uncertainty summaries, calibration, and selective prediction. Reported both prediction performance and how often the system accepts a decision.
- **Psychological Interpretation:** Analyzed self-report scores separately from training, including exploratory associations with physiological features and held-out model outputs.
- **Explainability:** Implemented Monte Carlo Shapley attribution, feature-family and modality summaries, beeswarm and waterfall plots, PDP, and first-order ALE for saved LOSO models.
- **Benchmark Dashboard:** Recomputed metrics from saved out-of-fold predictions, checked sample compatibility, identified duplicate runs, and generated comparative tables and figures.

---

## Dataset & Prediction Tasks

WESAD contains recordings from **15 participants** in a laboratory study, with chest- and wrist-worn sensors and accompanying self-reports. The recordings include cardiac/pulse activity, EDA, respiration, temperature, acceleration, and EMG. See the [original study](https://doi.org/10.1145/3242969.3242985) and [dataset documentation](https://doi.org/10.24432/C57K5T).

| Analysis | Target | Role in this project |
|---|---|---|
| Three-class classification | Baseline / Stress / Amusement | Original primary task |
| Stress vs. non-stress | Stress / Baseline + Amusement | Secondary |
| Baseline vs. stress | Baseline / Stress | Secondary |
| Four-class classification | Baseline / Stress / Amusement / Meditation | Exploratory |

**Tasks are benchmarked separately.** A binary score cannot be ranked directly against a three- or four-class score. Experimental condition labels are not continuous measurements of perceived stress.

## Evaluation Design

- **60-second windows with 50% overlap**, without crossing condition boundaries.
- **Outer LOSO:** each participant is held out once for testing.
- **Inner participant-grouped cross-validation:** hyperparameter tuning remains inside the training participants.
- **Training-only preprocessing:** imputation and scaling are fitted without the test participant.
- **Matched sensor comparisons:** the primary wrist/chest/fusion comparison uses identical common valid windows.
- **Primary endpoint:** balanced accuracy averaged equally across participants.
- **Additional metrics:** macro precision, recall, F1, per-class recall, ROC AUC, average precision, and confusion matrices.

Bootstrap intervals and paired comparisons use participants as the resampling unit, conditional on the fitted LOSO predictions. They do not include retraining inside each bootstrap replicate. Overlapping windows are not treated as independent participants.

---

## Results: Performance Without a Single “Winner” Narrative

The following values come from the saved OOF benchmark. Balanced accuracy and macro F1 are **equal-participant means**.

### Original Primary Comparisons

| Configuration | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Wrist — Random Forest | **68.29%** | **0.656** |
| Early fusion — Logistic Regression | 68.21% | 0.636 |

Wrist Random Forest achieved the highest observed mean among the original primary configurations, but its proximity to early-fusion logistic regression does not establish a decisive winner.

### Exploratory Three-Class Candidates

| Configuration | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Late fusion — Logistic Regression | **73.32%** | 0.700 |
| Early fusion — Random Forest, correlation filter 0.90 | **72.36%** | **0.704** |

The reduced Random Forest used approximately **35 rather than 45 features** and improved mean balanced accuracy by **4.92 percentage points** over its complete-feature counterpart. This is a post-hoc sensitivity result, not a replacement of the original primary analysis.

Late fusion exceeded the reduced Random Forest by approximately **0.96 percentage points**, with a conditional paired bootstrap interval of **−6.72 to +7.46 points**. The available comparison does not establish superiority between these candidates. Their relative position also depends on whether balanced accuracy or macro F1 is considered.

### Secondary Binary Tasks

| Task | Configuration | Balanced accuracy | Macro F1 |
|---|---|---:|---:|
| Stress vs. non-stress | Wrist — Logistic Regression | **90.33%** | 0.889 |
| Baseline vs. stress | Wrist — Logistic Regression | **91.22%** | 0.904 |

These results suggest that separating the laboratory stress condition is easier than distinguishing all three primary conditions. They do not demonstrate everyday stress detection or clinical validity.

### Benchmark at a Glance

The benchmark workflow generates:

- Side-by-side **precision, recall, and F1** comparisons.
- Subject-level performance and per-class recall heatmaps.
- Balanced-accuracy rankings with conditional intervals.
- Full-versus-reduced comparisons at **0.95, 0.90, and 0.80**.
- Performance versus feature-count plots.
- Paired comparisons restricted to matching tasks, participants, and windows.

The current benchmark audit validated **84 distinct configurations** and recognized **15 identical OOF reruns** as duplicates rather than independent evidence. The complete tables and decision rules are available in `artifacts/revision_v2/benchmark/` after execution.

---

## Explainability: Contributions, Not Causes

The all-experiment interpretation notebook explains **saved models on held-out participants**. It supports primary and secondary classifiers, advanced algorithms, fusion predictors, and saved reduced variants.

To avoid the SHAP–Numba compatibility issue encountered in the working environment, the implementation uses **Monte Carlo estimates of Shapley values through antithetic feature-order sampling**, without importing SHAP or Numba. It is **not exact TreeSHAP**.

- **Common output scale:** class probabilities across algorithms in the all-experiment notebook.
- **Training-only references:** background samples exclude the test participant.
- **Validation:** loaded models must reproduce their saved OOF probabilities.
- **Approximation diagnostics:** additivity checks and Monte Carlo standard errors.
- **Interpretation levels:** individual features, physiological families, and wrist/chest contributions within fused models.
- **Visual outputs:** global importance, beeswarm, individual waterfall, PDP, and first-order ALE plots.

Additivity does not establish convergence. Correlated inputs may distribute attribution across features, and hybrid samples may fall outside realistic physiological combinations. No contribution is interpreted as a causal effect of a physiological variable on stress.

## Reliability & the Individual Participant

The uncertainty workflow separates model-fitting, calibration, and test participants. It reports fold-specific thresholds, accepted-window counts, class-specific acceptance, and performance before and after rejection.

This distinction matters: a participant with 100% accuracy on two accepted windows does not have a reliable all-condition classifier. Some participants lose entire classes after rejection, while others retain confidently incorrect predictions.

Calibration and conformal analyses remain exploratory. Neither high confidence nor empirical coverage establishes guaranteed reliability for a new individual.

## Self-Reports: Connecting Predictions With Experience

Questionnaires remain **outside the classifier-training loop**. The workflow describes self-reports by condition and explores associations with physiological features and OOF model scores.

Model outputs are aggregated to participant–condition level before association analysis. Spearman correlations, multiple-comparison correction, paired baseline–TSST comparisons, and participant-sensitivity checks are available in the dedicated notebook.

Missing items remain missing; composite scores requiring those items are not replaced by incomplete sums. Questionnaire scoring and item mappings must be checked against the original documentation before psychological interpretation. Associations are exploratory and do not establish ecological or clinical validation.

---

## Notebook Guide

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
| `10_all_loso_experiments_interpretability_no_numba.ipynb` | Interpretation of all supported saved LOSO experiments |
| `10_training_block_permutation_loso.ipynb` | Separate conditional block-randomization diagnostic |
| `11_benchmark_and_model_comparison.ipynb` | Metric audit, visual benchmark, and explicit ranking rules |

The two notebook-10 files have different purposes; use their full filenames.

## Reproducibility & Statistical Boundaries

Configuration is stored in `configs/config.yaml`. The revised workflow records configuration and source-code SHA-256 fingerprints in `protocol_lock.json` and verifies them before using the frozen protocol. New source revisions must not be retroactively assigned to historical results.

Correlation filtering is a **post-hoc sensitivity analysis**, fitted within training folds, including inner validation. All thresholds are reported; none is promoted retrospectively to the primary model because it performs best.

The optional block-randomization notebook reassigns whole-episode labels **within each training participant**, leaving test labels untouched. It preserves episode counts by class but not necessarily window-level class proportions. Its p-value is conditional on an episode-exchangeability assumption; preserving temporal blocks does not establish that assumption. The implementation supports at least 1,000 repetitions with repeated tuning; implementation alone is not evidence that the complete run has finished.

## Getting Started

```bash
pip install -r requirements.txt

# Optional boosting models:
pip install -r requirements-exploratory.txt
```

Obtain WESAD separately and keep the extracted structure:

```text
data/
└── raw/
    └── WESAD/
        ├── S2/
        │   ├── S2.pkl
        │   └── S2_quest.csv
        └── ...
```

Review dataset and output settings in `configs/config.yaml`, then execute the core notebooks in order. The current revision uses `artifacts/revision_v2`; historical and synthetic outputs remain separate.

Interpretability requires persisted fitted models, fold metadata, and matching OOF predictions. Run `09b` if reduced models were not saved. Prediction CSVs alone cannot reconstruct a fitted classifier or calibrator. The interpretation notebook reports unsupported or unavailable experiments rather than silently training replacements.

Raw WESAD data are not distributed with the project archive. Obtain the dataset from its original distribution and follow its terms and acknowledgment requirements. Load pickle/joblib artifacts only from trusted sources.

## Tech Stack & Libraries

### Signal Processing & Statistical Analysis

- **Python, NumPy, Pandas:** signal tables, aggregation, and reproducible analysis.
- **SciPy:** filtering, peak detection, correlation, and statistical analysis.
- **PyArrow:** Parquet feature and attribution storage.

### Machine Learning & Interpretation

- **Scikit-Learn:** preprocessing pipelines, grouped validation, logistic regression, Random Forest, and MLP.
- **XGBoost & LightGBM:** exploratory gradient-boosting comparisons.
- **Joblib:** saved LOSO pipelines.
- **NumPy-based Shapley sampling:** dependency-independent attribution in the all-experiment interpretation notebook.

### Visualization & Reporting

- **Matplotlib & Seaborn:** distributions, model-comparison charts, attribution plots, and 300-dpi exports.
- **Jupyter:** executable analysis and reports.
- **Markdown, CSV, JSON:** inspection-ready results, manifests, and provenance.

## Limitations

- Fifteen participants and a laboratory protocol; no independent external validation.
- Condition labels, perceived stress, and peripheral physiology represent different measurements.
- Performance and prediction acceptance vary substantially across participants.
- Amusement remains difficult despite favorable aggregate scores.
- Overlapping windows and shared LOSO training sets limit independence.
- QC and EDA proxy features are engineering approximations, not validated physiological annotations.
- Model/threshold exploration introduces selection and multiplicity concerns.
- Monte Carlo attribution adds approximation error and is not a causal explanation.

*Research and methodological exploration only. The project does not provide a clinically validated diagnostic system or a validated real-world stress monitor.*

---

## References & Acknowledgments

This work uses the WESAD dataset introduced by:

**Schmidt, P., Reiss, A., Dürichen, R., Marberger, C., & Van Laerhoven, K. (2018).** Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. *Proceedings of the 20th ACM International Conference on Multimodal Interaction*, 400–408. [https://doi.org/10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)

**Schmidt, P., & Reiss, A. (2018).** WESAD (Wearable Stress and Affect Detection) [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C57K5T](https://doi.org/10.24432/C57K5T)

```bibtex
@inproceedings{schmidt2018wesad,
  author    = {Schmidt, Philip and Reiss, Attila and D{"u}richen, Robert
               and Marberger, Claus and Van Laerhoven, Kristof},
  title     = {Introducing WESAD, a Multimodal Dataset for Wearable
               Stress and Affect Detection},
  booktitle = {Proceedings of the 20th ACM International Conference
               on Multimodal Interaction},
  year      = {2018},
  pages     = {400--408},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3242969.3242985}
}

@misc{schmidt2018wesad_data,
  author    = {Schmidt, Philip and Reiss, Attila},
  title     = {WESAD (Wearable Stress and Affect Detection)},
  year      = {2018},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.24432/C57K5T}
}
```

The dataset and original paper remain the work of their respective authors. Results here describe this project's processing and evaluation choices and should not be presented as a direct improvement over the original paper without matching its protocol and metrics.
