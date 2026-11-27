#!/usr/bin/env python
"""Leakage-free external benchmark runner for HRC.

FalseQA is used as a binary false-premise classification task. Calibration is
fit only on a stratified sample from ``valid.csv`` (or ``train.csv``), while
metrics are reported on the disjoint ``test.csv`` split. Model and dataset
imports are deliberately lazy so ``--help`` remains quick and download-free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


FALSEQA_BASE = "https://raw.githubusercontent.com/thunlp/FalseQA/main"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the installed HRC cascade on FalseQA or TruthfulQA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("falseqa", "truthfulqa"), default="falseqa")
    parser.add_argument("--falseqa-dir", type=Path, help="Use local valid/train/test CSV files")
    parser.add_argument("--limit", type=int, help="Stratified evaluation sample size")
    parser.add_argument("--cal-size", type=int, default=60, help="FalseQA calibration sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Shared Stage-1 / Stage-2 model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-false-halt-rate", type=float, default=0.05,
                        help="Maximum calibration true-premise halt rate for v2_cal")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the model in 4-bit (requires CUDA and bitsandbytes)")
    parser.add_argument("--out-dir", type=Path, default=Path(".hrc_cache/external_bench"))
    parser.add_argument("--resume", action="store_true", help="Resume from the matching JSONL checkpoint")
    return parser


def normalize_question(text: str) -> str:
    """Normalize whitespace/case/punctuation for conservative split checks."""
    value = str(text).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^\w\s]", "", value, flags=re.UNICODE)


def assert_disjoint_questions(cal_rows: Sequence[dict], eval_rows: Sequence[dict]) -> None:
    cal_questions = {normalize_question(row["question"]) for row in cal_rows}
    eval_questions = {normalize_question(row["question"]) for row in eval_rows}
    overlap = cal_questions & eval_questions
    if overlap:
        examples = "; ".join(sorted(overlap)[:3])
        raise ValueError(f"Calibration/evaluation question overlap detected ({len(overlap)}): {examples}")


def stratified_sample(rows: Sequence[dict], limit: Optional[int], seed: int = 42) -> list[dict]:
    """Return a deterministic, approximately balanced label sample."""
    materialized = list(rows)
    if limit is None or limit >= len(materialized):
        return materialized
    if limit < 1:
        raise ValueError("limit must be positive")
    positives = [row for row in materialized if bool(row["is_false_premise"])]
    negatives = [row for row in materialized if not bool(row["is_false_premise"])]
    if limit > len(positives) + len(negatives):
        raise ValueError("limit exceeds available rows")
    rng = random.Random(seed)
    n_pos = limit // 2
    n_neg = limit // 2
    if limit % 2:
        if rng.random() < 0.5:
            n_pos += 1
        else:
            n_neg += 1
    if n_pos > len(positives) or n_neg > len(negatives):
        # Reallocate a missing class's quota to the class with remaining rows.
        deficit = max(0, n_pos - len(positives)) + max(0, n_neg - len(negatives))
        n_pos = min(n_pos, len(positives))
        n_neg = min(n_neg, len(negatives))
        if len(positives) - n_pos >= deficit:
            n_pos += deficit
        elif len(negatives) - n_neg >= deficit:
            n_neg += deficit
        else:
            raise ValueError("cannot draw requested sample from available class counts")
    selected = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    rng.shuffle(selected)
    return selected


def _label(value: Any) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "false premise", "false", "yes"}:
        return True
    if normalized in {"0", "false", "true premise", "valid", "no"}:
        return False
    raise ValueError(f"Unsupported FalseQA label: {value!r}; expected 0 or 1")


def _read_falseqa_csv(path: Path, split: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"FalseQA CSV has no header: {path}")
        fields = {field.strip().casefold(): field for field in reader.fieldnames}
        question_field = next((fields[name] for name in ("question", "query", "text", "prompt") if name in fields), None)
        label_field = fields.get("label")
        if question_field is None or label_field is None:
            raise ValueError(f"Expected question and label columns in {path}; got {reader.fieldnames}")
        id_field = fields.get("id")
        rows = []
        for index, raw in enumerate(reader):
            question = (raw.get(question_field) or "").strip()
            if not question:
                continue
            rows.append({
                "id": str(raw.get(id_field) or index) if id_field else str(index),
                "question": question,
                "label": int(_label(raw.get(label_field))),
                "is_false_premise": _label(raw.get(label_field)),
                "split": split,
            })
    return rows


def _download_falseqa(split: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{split}.csv"
    if target.is_file() and target.stat().st_size:
        return target
    errors = []
    for subdir in ("dataset", "data"):
        url = f"{FALSEQA_BASE}/{subdir}/{split}.csv"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if not data:
                raise ValueError("empty response")
            temporary = target.with_suffix(".csv.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
            return target
        except (OSError, urllib.error.URLError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download FalseQA split. Tried: " + " | ".join(errors))


def load_falseqa_split(split: str, *, cache_dir: Path, falseqa_dir: Optional[Path] = None) -> list[dict]:
    if falseqa_dir is not None:
        local = falseqa_dir / f"{split}.csv"
        if local.is_file():
            return _read_falseqa_csv(local, split)
    path = _download_falseqa(split, cache_dir)
    return _read_falseqa_csv(path, split)


def load_truthfulqa() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("TruthfulQA requires the optional 'datasets' package (pip install hrc-control[bench]).") from exc
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rows = []
    for index, item in enumerate(dataset):
        question = str(item.get("question", "")).strip()
        if question:
            rows.append({"id": str(index), "question": question, "is_false_premise": None,
                         "label": None, "split": "truthfulqa_generation"})
    return rows


def _average_precision(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Average precision for positive scores; ties are handled as one threshold."""
    if not positive_scores or not negative_scores:
        return float("nan")
    items = sorted(
        [(float(score), True) for score in positive_scores]
        + [(float(score), False) for score in negative_scores],
        key=lambda item: item[0], reverse=True,
    )
    total_positive = len(positive_scores)
    seen_positive = seen = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    while index < len(items):
        score = items[index][0]
        end = index
        while end < len(items) and items[end][0] == score:
            seen += 1
            seen_positive += int(items[end][1])
            end += 1
        recall = seen_positive / total_positive
        precision = seen_positive / seen
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return ap


def _classification_metrics(rows: Sequence[dict], cutoff: float) -> dict:
    labeled = [row for row in rows if row.get("is_false_premise") is not None]
    if not labeled:
        return {key: None for key in ("false_premise_halt_rate", "true_premise_false_halt_rate", "balanced_accuracy", "f1", "precision", "recall", "mean_latency_ms", "tp", "fp", "tn", "fn")}
    tp = fp = tn = fn = 0
    for row in labeled:
        predicted_false = bool(row.get("stage1_halted", False)) or float(row["s_dual"]) < cutoff
        is_false = bool(row["is_false_premise"])
        if predicted_false and is_false:
            tp += 1
        elif predicted_false:
            fp += 1
        elif is_false:
            fn += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced = (tpr + (1.0 - fpr)) / 2 if math.isfinite(tpr) and math.isfinite(fpr) else float("nan")
    return {
        "false_premise_halt_rate": tpr,
        "true_premise_false_halt_rate": fpr,
        "balanced_accuracy": balanced,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mean_latency_ms": sum(float(row.get("latency_ms", 0.0)) for row in labeled) / len(labeled),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _fingerprint(args: argparse.Namespace, cal_rows: Sequence[dict], eval_rows: Sequence[dict]) -> str:
    payload = {
        "dataset": args.dataset, "model": args.model, "device": args.device, "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens, "cal_size": args.cal_size, "seed": args.seed,
        "limit": args.limit,
        "cal_ids": [row["id"] for row in cal_rows], "eval_ids": [row["id"] for row in eval_rows],
        "cal_questions": [normalize_question(row["question"]) for row in cal_rows],
        "eval_questions": [normalize_question(row["question"]) for row in eval_rows],
        "cal_labels": [row.get("label") for row in cal_rows],
        "eval_labels": [row.get("label") for row in eval_rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_checkpoint(path: Path, *, resume: bool, fingerprint: str) -> dict[str, dict]:
    if not resume or not path.exists():
        return {}
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL checkpoint at {path}:{line_number}") from exc
            if row.get("_meta"):
                if row["_meta"].get("fingerprint") != fingerprint:
                    raise ValueError("Checkpoint fingerprint does not match this run; choose another --out-dir or omit --resume")
            elif row.get("example_key"):
                records[row["example_key"]] = row
    return records


def _append_checkpoint(path: Path, row: dict, *, write_header: bool, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if write_header:
            handle.write(json.dumps({"_meta": {"fingerprint": fingerprint, "format": 1}}, sort_keys=True) + "\n")
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _load_pipeline(args: argparse.Namespace):
    from hrc import HRCPipeline

    try:
        return HRCPipeline.from_pretrained(
            stage1_model=args.model,
            stage2_model=args.model,
            device=args.device,
            dtype=args.dtype,
            vm_threshold=-66.0,
            stage2_mode="dual",
            stage2_cutoff=0.0,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
        )
    except ImportError as exc:
        if args.load_in_4bit:
            raise RuntimeError("4-bit loading requires a CUDA runtime and the optional bitsandbytes package. Install bitsandbytes in a supported CUDA environment.") from exc
        raise


def _score_example(pipeline: Any, row: dict, split: str, fingerprint: str) -> dict:
    started = time.perf_counter()
    result = pipeline.query(row["question"], use_memory=False, run_stage2=False)
    # Stage-2 scoring is delegated to the installed pipeline's implementation.
    score, details = pipeline._score_stage2(row["question"], result.answer)
    stage1_halted = result.decision == "HALT_STAGE1"
    threshold = float(pipeline.config.stage2_cutoff)
    record = {
        "example_key": f"{split}:{row['id']}",
        "id": str(row["id"]), "split": split, "question": row["question"],
        "is_false_premise": row.get("is_false_premise"), "label": row.get("label"),
        "s_dual": float(score), "stage1_answer": result.answer,
        "stage1_raw_answer": result.raw_answer, "stage1_decision": result.decision,
        "stage1_halted": stage1_halted,
        "stage1_self_refusal": _explicitly_rejects(result.answer),
        "stage2_verdict": "HALT" if float(score) < threshold else "ACCEPT",
        "stage2_cutoff": threshold,
        "stage2_details": {key: float(value) for key, value in details.items() if isinstance(value, (int, float))},
        "latency_ms": max(float(result.latency_ms), (time.perf_counter() - started) * 1000.0),
        "run_fingerprint": fingerprint,
    }
    return record


def _explicitly_rejects(answer: str) -> bool:
    # Reuse the repository's existing Stage-1 refusal recognizer.
    from bench_llm_hrc import explicitly_rejects_premise
    return bool(explicitly_rejects_premise(answer))


def _score_rows(pipeline: Any, rows: Sequence[dict], split: str, checkpoint_path: Path,
                checkpoint: dict[str, dict], fingerprint: str) -> list[dict]:
    completed = []
    for row in rows:
        key = f"{split}:{row['id']}"
        if key in checkpoint:
            completed.append(checkpoint[key])
            continue
        scored = _score_example(pipeline, row, split, fingerprint)
        _append_checkpoint(checkpoint_path, scored, write_header=not checkpoint and not checkpoint_path.exists(), fingerprint=fingerprint)
        checkpoint[key] = scored
        completed.append(scored)
        print(f"[{split}] {len(completed)}/{len(rows)} {row['id']}: s_dual={scored['s_dual']:+.4f} {scored['stage1_decision']}", flush=True)
    return completed


def _rewrite_checkpoint(path: Path, checkpoint: dict[str, dict], fingerprint: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"_meta": {"fingerprint": fingerprint, "format": 1}}, sort_keys=True) + "\n")
        for key in sorted(checkpoint):
            handle.write(json.dumps(checkpoint[key], ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
    temporary.replace(path)


def _operating_summary(rows: Sequence[dict], cutoff: float) -> dict:
    return _classification_metrics(rows, cutoff)


def _summary_payload(args: argparse.Namespace, cal_rows: Sequence[dict], eval_rows: Sequence[dict],
                     cal_scored: Sequence[dict], eval_scored: Sequence[dict], cutoff: Optional[float],
                     calibration: Optional[dict], checkpoint_path: Path) -> dict:
    if args.dataset == "truthfulqa":
        discrimination = {"auroc": None, "average_precision": None, "note": "TruthfulQA generation has no false-premise labels."}
    else:
        false_scores = [-float(row["s_dual"]) for row in eval_scored if row["is_false_premise"]]
        true_scores = [-float(row["s_dual"]) for row in eval_scored if not row["is_false_premise"]]
        from bench_llm_hrc import compute_auroc
        discrimination = {
            "auroc": compute_auroc(false_scores, true_scores),
            "average_precision": _average_precision(false_scores, true_scores),
        }
    stage1_refusals = [row for row in eval_scored if row["stage1_self_refusal"]]
    baseline = {
        "self_refusal_rate": len(stage1_refusals) / len(eval_scored) if eval_scored else None,
        "false_premise_self_refusal_rate": (
            sum(bool(row["stage1_self_refusal"]) for row in eval_scored if row.get("is_false_premise") is True)
            / sum(row.get("is_false_premise") is True for row in eval_scored)
            if any(row.get("is_false_premise") is True for row in eval_scored) else None
        ),
    }
    default_metrics = _operating_summary(eval_scored, 0.0)
    calibrated_metrics = _operating_summary(eval_scored, cutoff) if cutoff is not None else None
    return {
        "protocol": (
            "FalseQA leakage-free disjoint valid/train calibration and test evaluation"
            if args.dataset == "falseqa"
            else "TruthfulQA generation split; no false-premise labels available"
        ),
        "dataset": args.dataset,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "calibration_n": len(cal_rows),
        "evaluation_n": len(eval_rows),
        "calibration_source": cal_rows[0]["split"] if cal_rows else None,
        "calibration": calibration,
        "discrimination": discrimination,
        "stage1_self_refusal_baseline": baseline,
        "operating_points": {
            "default_uncalibrated_cutoff_0": {"cutoff": 0.0, **default_metrics},
            "v2_cal": {"cutoff": cutoff, **calibrated_metrics} if calibrated_metrics is not None else None,
        },
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in eval_scored) / len(eval_scored) if eval_scored else None,
        "checkpoint": str(checkpoint_path),
    }


def _write_summaries(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _json_safe(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for name, metrics in summary["operating_points"].items():
        if metrics is None:
            rows.append((name, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"))
        else:
            rows.append((name, f"{metrics['cutoff']:.6g}", _fmt(metrics["false_premise_halt_rate"]),
                         _fmt(metrics["true_premise_false_halt_rate"]), _fmt(metrics["balanced_accuracy"]),
                         _fmt(metrics["f1"]), _fmt(metrics["precision"]), _fmt(metrics["recall"]),
                         _fmt(metrics["mean_latency_ms"])))
    table = [
        "| Operating point | Cutoff | False-premise HALT (TPR) | True-premise false HALT (FPR) | Balanced accuracy | F1 | Precision | Recall | Mean latency (ms/item) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    discr = summary["discrimination"]
    extra = "" if discr["auroc"] is None else f"\nScore discrimination: AUROC **{_fmt(discr['auroc'])}**, Average Precision **{_fmt(discr['average_precision'])}**."
    if discr["auroc"] is None:
        extra = "\nScore discrimination is unavailable because this dataset has no false-premise labels."
    content = (
        f"# External benchmark: {summary['dataset']}\n\n"
        f"Model: `{summary['model']}`; evaluation examples: {summary['evaluation_n']}; "
        f"calibration examples: {summary['calibration_n']}.\n\n"
        + "\n".join(table)
        + extra
        + f"\n\nNaive Stage-1 self-refusal rate: **{_fmt(summary['stage1_self_refusal_baseline']['self_refusal_rate'])}** overall "
        + f"(**{_fmt(summary['stage1_self_refusal_baseline']['false_premise_self_refusal_rate'])}** on false-premise items). "
        + f"Mean end-to-end latency: **{_fmt(summary['mean_latency_ms'])} ms/item**.\n"
    )
    (out_dir / "summary.md").write_text(content, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4f}" if math.isfinite(number) else "n/a"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_benchmark(args: argparse.Namespace, *, pipeline: Any = None) -> dict:
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if not 0.0 <= args.max_false_halt_rate <= 1.0:
        raise ValueError("max-false-halt-rate must be between 0 and 1")
    if args.cal_size < 2 and args.dataset == "falseqa":
        raise ValueError("cal-size must be at least 2")
    if args.dataset == "falseqa":
        cache_dir = Path(".hrc_cache/falseqa")
        eval_all = load_falseqa_split("test", cache_dir=cache_dir, falseqa_dir=args.falseqa_dir)
        cal_split = "valid"
        cal_path = args.falseqa_dir / "valid.csv" if args.falseqa_dir else cache_dir / "valid.csv"
        if args.falseqa_dir is not None and not cal_path.is_file():
            cal_split = "train"
        elif args.falseqa_dir is None and not cal_path.is_file():
            # Try valid first; only use train if the valid split truly is absent.
            try:
                cal_all = load_falseqa_split("valid", cache_dir=cache_dir)
            except RuntimeError:
                cal_split = "train"
            else:
                cal_rows = stratified_sample(cal_all, args.cal_size, args.seed)
        if "cal_rows" not in locals():
            cal_all = load_falseqa_split(cal_split, cache_dir=cache_dir, falseqa_dir=args.falseqa_dir)
            cal_rows = stratified_sample(cal_all, args.cal_size, args.seed)
        eval_rows = stratified_sample(eval_all, args.limit, args.seed + 1)
        assert_disjoint_questions(cal_all, eval_all)
        assert_disjoint_questions(cal_rows, eval_rows)
    else:
        cal_all, cal_rows = [], []
        eval_rows = load_truthfulqa()
        if args.limit is not None:
            rng = random.Random(args.seed)
            eval_rows = rng.sample(eval_rows, min(args.limit, len(eval_rows)))
    fingerprint = _fingerprint(args, cal_rows, eval_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / f"{args.dataset}_{fingerprint[:12]}.jsonl"
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = _read_checkpoint(checkpoint_path, resume=args.resume, fingerprint=fingerprint)
    all_expected_keys = [f"cal:{row['id']}" for row in cal_rows] + [f"eval:{row['id']}" for row in eval_rows]
    complete_checkpoint = bool(all_expected_keys) and all(key in checkpoint for key in all_expected_keys)
    if pipeline is None and not complete_checkpoint:
        pipeline = _load_pipeline(args)
    # Ensure fixed v1 defaults for evaluation and let v2_cal update only the S2 cutoff.
    if pipeline is not None:
        pipeline.config.stage2_mode = "dual"
        pipeline.config.stage2_cutoff = 0.0
    cal_scored = _score_rows(pipeline, cal_rows, "cal", checkpoint_path, checkpoint, fingerprint)
    cutoff = None
    calibration = None
    if args.dataset == "falseqa":
        # Use the repository's frozen v2_cal fitter, on Stage-1 passers only.
        from bench_llm_hrc import _fit_v2_calibration_cutoff
        candidates = [
            {"is_valid": not bool(row["is_false_premise"]), "s2_score": float(row["s_dual"])}
            for row in cal_scored if not row["stage1_halted"]
        ]
        valid_passers = sum(bool(row["is_valid"]) for row in candidates)
        max_valid_halts = math.floor(args.max_false_halt_rate * valid_passers + 1e-12)
        cutoff, rule, margin = _fit_v2_calibration_cutoff(
            candidates, score_key="s2_score", max_valid_halts=max_valid_halts
        )
        calibration = {
            "rule": rule, "margin": float(margin), "max_valid_halts": max_valid_halts,
            "stage1_passer_count": len(candidates), "valid_passer_count": valid_passers,
            "trap_passer_count": sum(not row["is_valid"] for row in candidates),
        }
        if pipeline is not None:
            pipeline.config.stage2_cutoff = float(cutoff)
    if pipeline is None and any(f"eval:{row['id']}" not in checkpoint for row in eval_rows):
        pipeline = _load_pipeline(args)
    if pipeline is not None and cutoff is not None:
        pipeline.config.stage2_cutoff = float(cutoff)
    eval_scored = _score_rows(pipeline, eval_rows, "eval", checkpoint_path, checkpoint, fingerprint)
    # Store an explicit Stage-2 verdict at the actual calibrated operating point.
    final_cutoff = float(cutoff) if cutoff is not None else 0.0
    for row in cal_scored + eval_scored:
        row["stage2_cutoff"] = final_cutoff
        row["stage2_verdict"] = "HALT" if float(row["s_dual"]) < final_cutoff else "ACCEPT"
        checkpoint[row["example_key"]] = row
    _rewrite_checkpoint(checkpoint_path, checkpoint, fingerprint)
    summary = _summary_payload(args, cal_rows, eval_rows, cal_scored, eval_scored, cutoff, calibration, checkpoint_path)
    _write_summaries(args.out_dir, summary)
    print(f"Wrote {args.out_dir / 'summary.json'} and {args.out_dir / 'summary.md'}", flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_benchmark(args)
    except Exception as exc:
        print(f"bench_external: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
