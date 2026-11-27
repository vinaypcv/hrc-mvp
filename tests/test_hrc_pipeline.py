"""CPU-only, download-free tests for the public package wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from hrc import HRCConfig, HRCResult, HRCPipeline, MemoryRecord


class StubEngine:
    def __init__(self, *, vm=-80.0, answer="baseline", memory_aware=False):
        self.vm = vm
        self.answer = answer
        self.memory_aware = memory_aware
        self.calls = 0

    def embed_texts(self, texts):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0] for _ in texts])

    def infer_with_vm(self, query, *, memory=None, v_crit=-66.0, **kwargs):
        self.calls += 1
        retrieved = memory.retrieve(self.embed_texts([query])[0], query) if memory else {
            "injected": False, "fact": None, "slot_id": None, "score": 0.0,
            "entity_match": True,
        }
        answer = retrieved["fact"] if self.memory_aware and retrieved["injected"] else self.answer
        return {
            "answer": answer,
            "raw_answer": answer,
            "v_m": self.vm,
            "halt": self.vm > v_crit,
            "res_injected": retrieved["injected"],
            "res_id": retrieved["slot_id"] if retrieved["injected"] else None,
            "res_entity_match": retrieved["entity_match"],
            "res_sim": retrieved["score"],
            "trajectory_answers": [answer] * 4,
        }


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, *, vm=-80.0, score=0.5, mode="dual", cutoff=0.0, **kwargs):
        engine = kwargs.pop("engine", None) or StubEngine(vm=vm)
        scorer = kwargs.pop("stage2_scorer", lambda _mode, _query, _answer: score)
        return HRCPipeline(
            HRCConfig(stage1_model="stub", stage2_mode=mode, stage2_cutoff=cutoff),
            stage1_engine=engine,
            stage2_engine=engine,
            stage2_scorer=scorer,
            **kwargs,
        )

    def test_cascade_routes_accept_and_both_halts(self):
        accepted = self.make_pipeline().query("valid question")
        self.assertEqual(accepted.decision, "ACCEPT")
        self.assertFalse(accepted.halted)

        stage1_halt = self.make_pipeline(vm=-65.0).query("uncertain question")
        self.assertEqual(stage1_halt.decision, "HALT_STAGE1")
        self.assertIsNone(stage1_halt.stage2_score)

        stage2_halt = self.make_pipeline(score=-0.1).query("false-premise question")
        self.assertEqual(stage2_halt.decision, "HALT_STAGE2")
        self.assertTrue(stage2_halt.halted)

    def test_stage2_off_and_explicit_bypass(self):
        pipeline = self.make_pipeline(
            mode="off",
            stage2_scorer=lambda *_args: self.fail("Stage 2 must be skipped"),
        )
        result = pipeline.query("question")
        self.assertEqual(result.decision, "ACCEPT")
        self.assertIsNone(result.stage2_score)

        pipeline = self.make_pipeline(score=-10.0)
        result = pipeline.query("question", run_stage2=False)
        self.assertEqual(result.decision, "ACCEPT")
        self.assertIsNone(result.stage2_score)

    def test_same_model_load_is_shared(self):
        loaded = object()
        with patch("hrc.pipeline._load_engine", return_value=loaded) as loader:
            pipeline = HRCPipeline.from_pretrained(
                "same/model", device="cpu", dtype="float32"
            )
        self.assertEqual(loader.call_count, 1)
        self.assertIs(pipeline.stage1_engine, pipeline.stage2_engine)
        self.assertIs(pipeline.stage1_engine, loaded)

    def test_layer3_guarded_injection_sibling_isolation_and_purge(self):
        engine = StubEngine(answer="unassisted", memory_aware=True)
        pipeline = self.make_pipeline(engine=engine)
        record = pipeline.memorize(
            "kestrel-fact", "Project Kestrel", "Kestrel project fact"
        )
        self.assertIsInstance(record, MemoryRecord)

        target = pipeline.query("What is the status of Project Kestrel?")
        self.assertTrue(target.memory_injected)
        self.assertEqual(target.memory_slot, "kestrel-fact")
        self.assertEqual(target.answer, "Kestrel project fact")

        sibling = pipeline.query("What is the status of Project Orion?")
        self.assertFalse(sibling.memory_injected)
        self.assertEqual(sibling.answer, "unassisted")

        purged = pipeline.purge("kestrel-fact")
        self.assertTrue(purged["purged"])
        self.assertLess(purged["purge_ms"], 1.0)
        self.assertTrue(purged["index_consistent"])
        reverted = pipeline.query("What is the status of Project Kestrel?")
        self.assertFalse(reverted.memory_injected)
        self.assertEqual(reverted.answer, "unassisted")

    def test_stage2_calibration_uses_max_rejection_then_margin(self):
        pipeline = self.make_pipeline()
        result = pipeline.calibrate_stage2(
            valid_scores=[0.6, 0.7, 0.8],
            trap_scores=[0.4, 0.5, 0.9],
            max_valid_halts=1,
        )
        self.assertEqual(
            set(result), {"cutoff", "margin", "valid_halts", "traps_rejected"}
        )
        self.assertEqual(result["valid_halts"], 1)
        self.assertEqual(result["traps_rejected"], 2)
        self.assertAlmostEqual(result["margin"], 0.2)
        self.assertAlmostEqual(pipeline.config.stage2_cutoff, result["cutoff"])

    def test_result_json_and_module_help(self):
        result = self.make_pipeline().query("a prompt")
        payload = json.dumps(result.to_dict())
        restored = json.loads(payload)
        self.assertEqual(restored["query"], "a prompt")
        self.assertIn(restored["decision"], {"ACCEPT", "HALT_STAGE1", "HALT_STAGE2"})
        self.assertIsInstance(result, HRCResult)

        process = subprocess.run(
            [sys.executable, "-m", "hrc", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        for option in ("--model", "--query", "--vm-threshold", "--stage2-mode", "--stage2-cutoff", "--memorize", "--json"):
            self.assertIn(option, process.stdout)


if __name__ == "__main__":
    unittest.main()
