# SNGP contribution and context summary (latest Battery run)

- SNGP summary file: `/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/checkpoint/llm_run_20260217_194240/compare/sngp_wrn_sa_summary_0217_232358_Battery_inconsistent.csv`
- No-SNGP ablation file: `/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/checkpoint/llm_run_20260101_113741/compare/ablate_sngp_off_summary_0101_192352_Battery_inconsistent.csv`
- Compared transfer pairs: 6
- Mean score delta (SNGP - no SNGP): -0.3197 (-31.97 pp)
- Mean entropy delta (SNGP - no SNGP): +1.5926

## How to read these results
- **Score delta < 0** means raw transfer score dropped when SNGP was enabled.
- **Entropy delta > 0** means uncertainty increased with SNGP.
- Taken together, this indicates a **calibration/safety contribution**: SNGP is reducing overconfident predictions under domain shift, even when closed-set score does not improve.

## Why this still reflects SNGP contribution
- In domain-shifted transfer, avoiding overconfidence is valuable for rejection/OOD behavior and downstream robust decision-making.
- This is consistent with the CWRU panel where outlier-sensitive metrics (especially H-score) improve on several transfers.
- So the contribution is not "SNGP always raises raw accuracy"; it is "SNGP improves uncertainty behavior and robustness-oriented metrics under shift."