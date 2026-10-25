# Methodology

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

## Reproducibility and statistical boundaries

Configuration is stored in `configs/config.yaml`. The revised workflow records configuration and source-code SHA-256 fingerprints in `protocol_lock.json` and verifies them before using the frozen protocol. New source revisions must not be retroactively assigned to historical results.

Correlation filtering is a **post-hoc sensitivity analysis**, fitted within training folds, including inner validation. All thresholds are reported; none is promoted retrospectively to the primary model because it performs best.

The optional block-randomization notebook reassigns whole-episode labels **within each training participant**, leaving test labels untouched. It preserves episode counts by class but not necessarily window-level class proportions. Its p-value is conditional on an episode-exchangeability assumption; preserving temporal blocks does not establish that assumption. The completed wrist logistic-regression diagnostic used 1,000 repetitions with repeated tuning. Observed balanced accuracy was 0.66108; none of the randomized scores reached it. The plus-one Monte Carlo p-value was 1/1001 (approximately 0.001). This is a conditional diagnostic for this configuration, not a test establishing a globally best model.
