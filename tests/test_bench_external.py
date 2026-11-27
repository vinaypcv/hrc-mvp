"""CPU-only tests for the resumable external benchmark runner."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hrc import HRCResult
from scripts import bench_external as bench


class StubPipeline:
    def __init__(self):
        self.config = SimpleNamespace(stage2_mode="dual", stage2_cutoff=0.0)
        self.query_calls = []
        self.scores = {}

    def query(self, question, *, use_memory=True, run_stage2=True):
        self.query_calls.append(question)
        answer = "There is no such place." if "impossible" in question.lower() else "A plausible answer."
        return HRCResult(
            query=question,
            answer=answer,
            raw_answer=answer,
            decision="HALT_STAGE1" if "uncertain" in question.lower() else "ACCEPT",
            halted="uncertain" in question.lower(),
            vm_mv=-50.0 if "uncertain" in question.lower() else -80.0,
            stage2_score=None,
            stage2_mode="dual",
            memory_injected=False,
            memory_slot=None,
            latency_ms=3.0,
            metadata={},
        )

    def _score_stage2(self, question, answer):
        return self.scores[question], {"d_pos": self.scores[question] + 0.1, "d_neg": -self.scores[question]}


class ExternalBenchmarkTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "question", "label"))
            writer.writeheader()
            writer.writerows(rows)

    def test_stratified_sampling_is_balanced_and_deterministic(self):
        rows = [
            {"id": str(index), "question": f"Question {index}", "is_false_premise": index % 2 == 0}
            for index in range(40)
        ]
        first = bench.stratified_sample(rows, 14, seed=7)
        second = bench.stratified_sample(rows, 14, seed=7)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(sum(row["is_false_premise"] for row in first), 7)
        self.assertEqual(sum(not row["is_false_premise"] for row in first), 7)

    def test_csv_mapping_and_strict_zero_overlap_assertion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_csv(root / "valid.csv", [
                {"id": "v1", "question": "A valid unique question?", "label": 0},
                {"id": "v2", "question": "A false unique question?", "label": 1},
            ])
            rows = bench.load_falseqa_split("valid", cache_dir=root / "cache", falseqa_dir=root)
            self.assertEqual([row["is_false_premise"] for row in rows], [False, True])
            bench.assert_disjoint_questions(rows, [{"question": "Different question?"}])
            with self.assertRaisesRegex(ValueError, "overlap"):
                bench.assert_disjoint_questions(rows, [{"question": "  A VALID UNIQUE QUESTION?  "}])

    def test_metric_math_auroc_tpr_fpr_and_f1(self):
        rows = [
            {"is_false_premise": True, "s_dual": -1.0, "stage1_halted": False, "latency_ms": 4.0},
            {"is_false_premise": True, "s_dual": 1.0, "stage1_halted": False, "latency_ms": 4.0},
            {"is_false_premise": False, "s_dual": 0.0, "stage1_halted": False, "latency_ms": 4.0},
            {"is_false_premise": False, "s_dual": 2.0, "stage1_halted": True, "latency_ms": 4.0},
        ]
        from bench_llm_hrc import compute_auroc
        auc = compute_auroc([-row["s_dual"] for row in rows if row["is_false_premise"]],
                            [-row["s_dual"] for row in rows if not row["is_false_premise"]])
        ap = bench._average_precision([1.0, -1.0], [0.0, -2.0])
        metrics = bench._classification_metrics(rows, 0.0)
        self.assertEqual(auc, 0.75)
        self.assertAlmostEqual(ap, 5 / 6)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["false_premise_halt_rate"], 0.5)
        self.assertEqual(metrics["true_premise_false_halt_rate"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_resume_skips_completed_jsonl_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {"id": "1", "question": "valid one", "is_false_premise": False, "label": 0},
                {"id": "2", "question": "false one", "is_false_premise": True, "label": 1},
            ]
            pipeline = StubPipeline()
            pipeline.scores = {"valid one": 0.8, "false one": -0.8}
            fingerprint = "fingerprint"
            checkpoint_path = root / "checkpoint.jsonl"
            checkpoint = {}
            first = bench._score_rows(pipeline, rows[:1], "cal", checkpoint_path, checkpoint, fingerprint)
            self.assertEqual(len(first), 1)
            loaded = bench._read_checkpoint(checkpoint_path, resume=True, fingerprint=fingerprint)
            before = len(pipeline.query_calls)
            result = bench._score_rows(pipeline, rows[:1], "cal", checkpoint_path, loaded, fingerprint)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(pipeline.query_calls), before)

    def test_end_to_end_synthetic_run_writes_summaries_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            cal = []
            evaluation = []
            for index in range(8):
                label = index % 2
                cal.append({"id": f"c{index}", "question": f"Calibration unique {index}?", "label": label})
            for index in range(12):
                label = index % 2
                evaluation.append({"id": f"e{index}", "question": f"Evaluation unique {index}?", "label": label})
            self._write_csv(source / "valid.csv", cal)
            self._write_csv(source / "test.csv", evaluation)
            args = bench._parser().parse_args([
                "--falseqa-dir", str(source), "--cal-size", "8", "--limit", "8",
                "--out-dir", str(root / "out"), "--max-false-halt-rate", "0.25",
            ])
            pipeline = StubPipeline()
            pipeline.scores = {
                row["question"]: (-0.8 if int(row["id"][1:]) % 2 else 0.8)
                for row in cal + evaluation
            }
            summary = bench.run_benchmark(args, pipeline=pipeline)
            self.assertEqual(summary["calibration_n"], 8)
            self.assertEqual(summary["evaluation_n"], 8)
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "summary.md").is_file())
            saved = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset"], "falseqa")
            self.assertIn("v2_cal", saved["operating_points"])
            checkpoint = Path(summary["checkpoint"])

            resumed_args = bench._parser().parse_args([
                "--falseqa-dir", str(source), "--cal-size", "8", "--limit", "8",
                "--out-dir", str(root / "out"), "--max-false-halt-rate", "0.25", "--resume",
            ])
            call_count = len(pipeline.query_calls)
            with patch.object(bench, "_load_pipeline", side_effect=AssertionError("complete resume must not load weights")):
                bench.run_benchmark(resumed_args)
            self.assertEqual(len(pipeline.query_calls), call_count)
            records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum("example_key" in row for row in records), 16)
            self.assertTrue(all("stage2_verdict" in row and "s_dual" in row for row in records if "example_key" in row))


if __name__ == "__main__":
    unittest.main()
