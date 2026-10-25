# Reliability and selective prediction

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
