# External benchmark: falseqa

Model: `Qwen/Qwen2.5-7B-Instruct`; evaluation examples: 200; calibration examples: 60.

| Operating point | Cutoff | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Precision | Recall | Mean latency (ms/item) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default_uncalibrated_cutoff_0 | 0 | 0.9600 | 0.6500 | 0.6550 | 0.7356 | 0.5963 | 0.9600 | 1766.0876 |
| v2_cal | -10.8203 | 0.6800 | 0.3300 | 0.6750 | 0.6766 | 0.6733 | 0.6800 | 1766.0876 |

## Decoupled diagnostic operating points

| Operating point | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Stealth AUROC (non-refused) |
|---|---:|---:|---:|---:|---:|
| stage1_vm_only | 0.3300 | 0.3000 | 0.5150 | 0.4049 | n/a |
| stage1_self_refusal_only | 0.3700 | 0.0600 | 0.6550 | 0.5175 | n/a |
| stage2_v2_cal_only | 0.5500 | 0.0300 | 0.7600 | 0.6962 | 0.8696 |
| combined_self_refusal_or_v2_cal | 0.6500 | 0.0800 | 0.7850 | 0.7514 | n/a |
Score discrimination: AUROC **0.8976**, Average Precision **0.8987**.

Naive Stage-1 self-refusal rate: **0.2150** overall (**0.3700** on false-premise items). Mean end-to-end latency: **1766.0876 ms/item**.
