# Colab Quickstart

Use a Colab runtime with a **T4 GPU** (Runtime -> Change runtime type -> T4 GPU). The benchmark resolves `--dtype auto` to float16 on T4. `requirements.txt` intentionally leaves PyTorch to Colab's preinstalled CUDA build.

## Setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content
!git clone https://github.com/vinaypcv/hrc-mvp.git
%cd /content/hrc-mvp
!pip install -r requirements.txt
```

Check that Colab attached the GPU:

```python
import torch
assert torch.cuda.is_available(), "Select a GPU runtime in Colab first"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
```

## Locked v1 Stage-1 run on T4

This runs the locked 40-query Phase 1 plus 27-query holdout with Qwen2.5-1.5B and writes its device-tagged cache to Drive so it survives a runtime reset.

```python
!mkdir -p /content/drive/MyDrive/hrc-mvp-cache
!python bench_llm_hrc.py --m2-l2-stage1-compare --stage1-model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype auto --stage1-cache-dir /content/drive/MyDrive/hrc-mvp-cache
```

## Resumable Stage-2 run

Run the dual-polarity verifier against that Stage-1 cache. Each completed query is saved atomically; rerun the same cell after an interruption to resume.

```python
!python bench_llm_hrc.py --m2-l2-asym --stage1-cache-model Qwen/Qwen2.5-1.5B-Instruct --stage1-cache-dtype float16 --stage1-cache-device cuda --stage1-cache-dir /content/drive/MyDrive/hrc-mvp-cache --stage2-model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype auto --stage2-dual-polarity --stage2-cache-dir /content/drive/MyDrive/hrc-mvp-cache
```
