# External benchmark: falseqa

Model: `Qwen/Qwen2.5-3B-Instruct`; evaluation examples: 200; calibration examples: 60.

| Operating point | Cutoff | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Precision | Recall | Mean latency (ms/item) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default_uncalibrated_cutoff_0 | 0 | 0.9900 | 0.8600 | 0.5650 | 0.6947 | 0.5351 | 0.9900 | 1309.8766 |
| v2_cal | -8.04688 | 0.5900 | 0.5400 | 0.5250 | 0.5540 | 0.5221 | 0.5900 | 1309.8766 |

## Decoupled diagnostic operating points

| Operating point | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Stealth AUROC (non-refused) |
|---|---:|---:|---:|---:|---:|
| stage1_vm_only | 0.5400 | 0.5000 | 0.5200 | 0.5294 | n/a |
| stage1_self_refusal_only | 0.3500 | 0.0500 | 0.6500 | 0.5000 | n/a |
| stage2_v2_cal_only | 0.1700 | 0.1200 | 0.5250 | 0.2636 | 0.6965 |
| combined_self_refusal_or_v2_cal | 0.4800 | 0.1700 | 0.6550 | 0.5818 | n/a |
Score discrimination: AUROC **0.7059**, Average Precision **0.6429**.

Naive Stage-1 self-refusal rate: **0.2000** overall (**0.3500** on false-premise items). Mean end-to-end latency: **1309.8766 ms/item**.
