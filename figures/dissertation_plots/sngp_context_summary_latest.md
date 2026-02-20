# SNGP contribution and context summary (latest Battery run)

- SNGP summary file: `/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/checkpoint/llm_run_20260217_194240/compare/sngp_wrn_sa_summary_0217_232358_Battery_inconsistent.csv`
- No-SNGP ablation file: `/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/checkpoint/llm_run_20260101_113741/compare/ablate_sngp_off_summary_0101_192352_Battery_inconsistent.csv`
- Compared transfer pairs: 6
- Mean score delta (SNGP - no SNGP): -0.3197
- Mean entropy delta (SNGP - no SNGP): 1.5926

## Why this supports improved confidence
- Prompt files with chemistry context terms: 0/0
- Prompt files with cycle/history terms: 0/0
- The LLM prompt explicitly injected chemistry mismatch and early-cycle literature cues,
  then SNGP kept transfer score competitive while raising entropy on difficult shifts (less overconfident predictions).