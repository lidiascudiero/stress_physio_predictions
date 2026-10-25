# Results

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
