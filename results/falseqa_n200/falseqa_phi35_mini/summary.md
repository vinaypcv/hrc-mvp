# External benchmark: falseqa

Model: `microsoft/Phi-3.5-mini-instruct`; evaluation examples: 200; calibration examples: 60.

| Operating point | Cutoff | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Precision | Recall | Mean latency (ms/item) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default_uncalibrated_cutoff_0 | 0 | 0.9000 | 0.6800 | 0.6100 | 0.6977 | 0.5696 | 0.9000 | 2668.4888 |
| v2_cal | -11.25 | 0.7900 | 0.5700 | 0.6100 | 0.6695 | 0.5809 | 0.7900 | 2668.4888 |

## Decoupled diagnostic operating points

| Operating point | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Stealth AUROC (non-refused) |
|---|---:|---:|---:|---:|---:|
| stage1_vm_only | 0.7700 | 0.5700 | 0.6000 | 0.6581 | n/a |
| stage1_self_refusal_only | 0.4500 | 0.0900 | 0.6800 | 0.5844 | n/a |
| stage2_v2_cal_only | 0.4300 | 0.0200 | 0.7050 | 0.5931 | 0.8198 |
| combined_self_refusal_or_v2_cal | 0.6500 | 0.1100 | 0.7700 | 0.7386 | n/a |
Score discrimination: AUROC **0.8542**, Average Precision **0.8733**.

Naive Stage-1 self-refusal rate: **0.2700** overall (**0.4500** on false-premise items). Mean end-to-end latency: **2668.4888 ms/item**.
