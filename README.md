 # Wearable stress and affect: from physiological signals to interpretable predictions

This repository implements a Machine Learning workflow for **subject-independent classification of laboratory stress and affect**, using wrist and chest recordings from the WESAD dataset.

The project connects physiological feature engineering, nested participant-level validation, multimodal fusion, uncertainty analysis, and model interpretation. The central question goes beyond which algorithm scores highest: **what information survives across people, how much does each sensor setup contribute, and where do the predictions become unreliable?**

This is an independent analysis of WESAD, not an official implementation or an exact reproduction of its original benchmark. Dataset collection and the experimental protocol belong to the original authors. [Original paper](https://doi.org/10.1145/3242969.3242985) · [Dataset](https://doi.org/10.24432/C57K5T)

---

## The “Neurocognitive integrity” angle

> **Author’s note:** My background in **Cognitive Neuroscience** shaped how I approached this project. I treated physiological signals, experimental conditions, and subjective experience as related but distinct sources of information. A participant exposed to a stress task is not automatically experiencing a uniform level of stress, and a confident algorithm is not automatically a reliable one. This perspective motivated participant-independent evaluation, separate self-report analyses, and an explicit examination of individual differences. My aim was to connect computational predictions with psychological interpretation while keeping the limits of each measurement visible. These are **peripheral physiological recordings not direct measurements of neural activity**.

---

## Project workflow

- **Signal audit and quality control:** loaded the extracted recordings, checked participant coverage, signal availability, missing values, and window-level quality. Added ACC cross-correlation as a diagnostic for possible temporal offsets, without automatically realigning recordings.
- **Feature engineering and exploration:** extracted cardiac, pulse, electrodermal, respiratory, temperature, movement, and muscle features. Explored distributions, condition differences, participant variability, outliers, and feature redundancy.
- **Subject-independent benchmarking:** compared majority classification, logistic regression, Random Forest, XGBoost, LightGBM, and MLP using LOSO evaluation and inner cross-validation grouped by participant.
- **Multimodal analysis:** compared wrist, chest, early fusion, late fusion, and simulated missing-modality scenarios. Primary sensor comparisons used the same common valid windows.
- **Robustness and reliability:** evaluated post-hoc correlation filtering, participant-level uncertainty summaries, calibration, and selective prediction. Reported both prediction performance and how often the system accepts a decision.
- **Psychological interpretation:** analyzed self-report scores separately from training, including exploratory associations with physiological features and held-out model outputs.
- **Explainability:** implemented Monte Carlo Shapley attribution, feature-family and modality summaries, beeswarm and waterfall plots, PDP, and first-order ALE for saved LOSO models.
- **Benchmark dashboard:** recomputed metrics from saved out-of-fold predictions, checked sample compatibility, identified duplicate runs, and generated comparative tables and figures.

---

## Dataset and prediction tasks

WESAD contains recordings from **15 participants** in a laboratory study, with chest- and wrist-worn sensors and accompanying self-reports. The recordings include cardiac/pulse activity, EDA, respiration, temperature, acceleration, and EMG. See the [original study](https://doi.org/10.1145/3242969.3242985) and [dataset documentation](https://doi.org/10.24432/C57K5T).

| Analysis | Target | Role in this project |
|---|---|---|
| Three-class classification | Baseline / Stress / Amusement | Original primary task |
| Stress vs. non-stress | Stress / Baseline + Amusement | Secondary |
| Baseline vs. stress | Baseline / Stress | Secondary |
| Four-class classification | Baseline / Stress / Amusement / Meditation | Exploratory |

**Tasks are benchmarked separately.** A binary score cannot be ranked directly against a three- or four-class score. Experimental condition labels are not continuous measurements of perceived stress.

## Evaluation design

- **60-second windows with 50% overlap**, without crossing condition boundaries.
- **Outer LOSO:** each participant is held out once for testing.
- **Inner participant-grouped cross-validation:** hyperparameter tuning remains inside the training participants.
- **Training-only preprocessing:** imputation and scaling are fitted without the test participant.
- **Matched sensor comparisons:** the primary wrist/chest/fusion comparison uses identical common valid windows.
- **Primary endpoint:** balanced accuracy averaged equally across participants.
- **Additional metrics:** macro precision, recall, F1, per-class recall, ROC AUC, average precision, and confusion matrices.

Bootstrap intervals and paired comparisons use participants as the resampling unit, conditional on the fitted LOSO predictions. They do not include retraining inside each bootstrap replicate. Overlapping windows are not treated as independent participants.

---

## Results

The following values come from the saved OOF benchmark. Balanced accuracy and macro F1 are **equal-participant means**.

### Original primary comparisons

| Configuration | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Wrist Random Forest | **68.29%** | **0.656** |
| Early fusion Logistic Regression | 68.21% | 0.636 |

Wrist Random Forest achieved the highest observed mean among the original primary configurations, but its proximity to early-fusion logistic regression does not establish a decisive winner.

### Exploratory three-class candidates

| Configuration | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Late fusion  Logistic Regression | **73.32%** | 0.700 |
| Early fusion  Random Forest, correlation filter 0.90 | **72.36%** | **0.704** |

The reduced Random Forest used approximately **35 rather than 45 features** and improved mean balanced accuracy by **4.92 percentage points** over its complete-feature counterpart. This is a post-hoc sensitivity result, not a replacement of the original primary analysis.

Late fusion exceeded the reduced Random Forest by approximately **0.96 percentage points**, with a conditional paired bootstrap interval of **−6.72 to +7.46 points**. The available comparison does not establish superiority between these candidates. Their relative position also depends on whether balanced accuracy or macro F1 is considered.

### Secondary binary tasks

| Task | Configuration | Balanced accuracy | Macro F1 |
|---|---|---:|---:|
| Stress vs. non-stress | Wrist  Logistic Regression | **90.33%** | 0.889 |
| Baseline vs. stress | Wrist  Logistic Regression | **91.22%** | 0.904 |

These results suggest that separating the laboratory stress condition is easier than distinguishing all three primary conditions. They do not demonstrate everyday stress detection or clinical validity.

### Benchmark at a Glance

The benchmark audit validated **84 distinct configurations** and recognized
**15 identical OOF reruns** as duplicates rather than independent evidence.

| Comparison | Key finding |
|---|---|
| Original primary models | Wrist Random Forest achieved **68.29% balanced accuracy** and **0.656 macro F1**. |
| Exploratory three-class models | Late fusion achieved **73.32% balanced accuracy** and **0.700 macro F1**. |
| Feature-redundancy sensitivity | Early-fusion Random Forest with a **0.90 correlation filter** reached **72.36% balanced accuracy** and **0.704 macro F1**, retaining approximately **35 rather than 45 features**. |
| Secondary stress/non-stress task | Wrist logistic regression reached **90.33% balanced accuracy** and **0.889 macro F1**. |

Late fusion exceeded the reduced early-fusion Random Forest by only
**0.96 percentage points**, with a conditional paired bootstrap interval
of **−6.72 to +7.46 points**. The comparison does not establish a definitive
winner.

The workflow provides side-by-side precision, recall and F1 charts,
participant-level heatmaps, class-specific results, sensitivity curves,
and paired comparisons restricted to matching tasks and windows.

Complete tables and decision rules are available in
`artifacts/revision_v2/benchmark/` after execution.

---

## Explainability: what do the models use?

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

## Reliability: better scores can hide limited availability

The reliability procedure uses **11 fitting participants, 3 calibration
participants, and 1 test participant** per LOSO fold.

It accepted **75.42% of windows overall**, but participant-level acceptance
ranged from **2.94% to 100%**. Class-specific acceptance was:

- **Stress:** 82.4%.
- **Baseline:** 74.2%.
- **Amusement:** 66.9%.

Among the **12 participants retaining all three classes after rejection**,
mean balanced accuracy increased only from **63.2% to 64.7%**.
This selected-subset result is not an improvement estimate for all
15 participants.

For S17, 100% accepted-window accuracy was based on only **two stress
windows**. Other participants retained confidently incorrect predictions.
The findings highlight the trade-off between accuracy, class coverage,
and availability of a decision.

In the separate exploratory conformal analysis, overall prediction-set
coverage was **89.75%**, but amusement coverage was only **63.13%**.
Overall coverage therefore concealed substantial class differences.

Calibration, abstention and conformal prediction remain exploratory;
none establishes guaranteed reliability for a new individual.

## Self-reports: connecting predictions with experience

Questionnaire processing covers **15 participants and 75 episodes**.
Self-reports remain **outside classifier training**.

Reported experience increased during the stress task:

| Measure | Baseline median | TSST median | Holm-adjusted paired-test p |
|---|---:|---:|---:|
| Stressed item | 2 | 4 | 0.0013 |
| STAI-6 keyed sum | 10 | 19 | 0.0013 |

However, a condition difference did not translate into clear evidence
that the classifier measured individual stress intensity.

During TSST, mean classifier stress probability had Spearman correlations
of **0.320 with the stressed item** and **0.221 with STAI**.
Both bootstrap intervals included zero, and none of the six model–self-report
associations remained significant after Holm correction.

Separately, physiological-feature analyses found descriptive TSST
correlations of **0.677 for heart rate** and **0.527 for pulse rate**
with the stressed item. These feature correlations did not include
inferential tests and should not be presented as statistically established
associations.

The distinction is central: **classifying an experimental condition,
describing physiological variation, and measuring subjective intensity
are different objectives**.

Missing questionnaire items are not imputed. Item mappings and scoring
must be checked against the original documentation. These exploratory
analyses do not establish causality, ecological validity, or clinical validity.

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

## Reproducibility and statistical boundaries

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

## Tech stack

### Signal Processing and statistical analysis

- **Python, NumPy, Pandas:** signal tables, aggregation, and reproducible analysis.
- **SciPy:** filtering, peak detection, correlation, and statistical analysis.
- **PyArrow:** Parquet feature and attribution storage.

### Machine Learning and interpretation

- **Scikit-Learn:** preprocessing pipelines, grouped validation, logistic regression, Random Forest, and MLP.
- **XGBoost and LightGBM:** exploratory gradient-boosting comparisons.
- **Joblib:** saved LOSO pipelines.
- **NumPy-based Shapley sampling:** dependency-independent attribution in the all-experiment interpretation notebook.

### Visualization and reporting

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

## References andacknowledgments

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
