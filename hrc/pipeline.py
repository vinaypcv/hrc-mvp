"""Public pipeline wrapper around the repository's validated HRC components.

The implementation delegates Layer 1/2 inference, Stage-2 scoring, and Layer 3
retrieval to ``bench_llm_hrc`` so the package does not fork the benchmark math.
Model loading is lazy: importing :mod:`hrc` does not load model weights.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class HRCConfig:
    stage1_model: str = DEFAULT_MODEL
    stage2_model: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    vm_threshold: float = -66.0
    stage2_mode: str = "dual"
    stage2_cutoff: float = 0.0
    enable_layer3: bool = True
    max_new_tokens: int = 64
    stage1_prompt_style: str = "validated-v1"
    stage2_prompt_style: str = "premise-audit-v1"

    def __post_init__(self) -> None:
        if self.stage2_model is None:
            self.stage2_model = self.stage1_model
        if self.stage2_mode not in {"dual", "direct", "off"}:
            raise ValueError("stage2_mode must be 'dual', 'direct', or 'off'")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be 'auto', 'float16', 'bfloat16', or 'float32'")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not self.stage1_model:
            raise ValueError("stage1_model must be non-empty")
        if self.stage1_prompt_style != "validated-v1":
            raise ValueError("stage1_prompt_style must be 'validated-v1'")
        if self.stage2_prompt_style != "premise-audit-v1":
            raise ValueError("stage2_prompt_style must be 'premise-audit-v1'")


@dataclass
class HRCResult:
    query: str
    answer: str
    raw_answer: str
    decision: str
    halted: bool
    vm_mv: float
    stage2_score: Optional[float]
    stage2_mode: str
    memory_injected: bool
    memory_slot: Optional[str]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this result."""
        return asdict(self)


@dataclass
class MemoryRecord:
    slot_id: str
    entity: str
    fact: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


def _bench_module():
    # Keep package import and CLI --help light; importing this legacy benchmark
    # also imports torch/transformers and configures its runtime environment.
    import bench_llm_hrc

    return bench_llm_hrc


def _load_engine(
    model_id: str,
    *,
    device: str,
    dtype: str,
    force_offline: bool = False,
    load_in_4bit: bool = False,
):
    bench = _bench_module()
    return bench.HRCLLMEngine(
        model_id=model_id,
        force_offline=force_offline,
        device=device,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )


class HRCPipeline:
    """Layer 3 retrieval -> validated V_m gate -> frozen Stage-2 audit cascade."""

    def __init__(
        self,
        config: Optional[HRCConfig] = None,
        *,
        stage1_engine: Any = None,
        stage2_engine: Any = None,
        stage2_scorer: Optional[Callable[..., Any]] = None,
        memory: Any = None,
        memory_tau_res: float = 0.45,
        memory_tau_inject: float = 0.78,
    ) -> None:
        self.config = config or HRCConfig()
        self.stage1_engine = stage1_engine
        self.stage2_engine = stage2_engine if stage2_engine is not None else stage1_engine
        self._stage2_scorer = stage2_scorer
        self._memory_tau_res = memory_tau_res
        self._memory_tau_inject = memory_tau_inject
        self.memory = memory if memory is not None else self._new_memory()
        self._memory_records: dict[str, MemoryRecord] = {}

    @classmethod
    def from_pretrained(
        cls,
        stage1_model: str = DEFAULT_MODEL,
        stage2_model: Optional[str] = None,
        **kwargs: Any,
    ) -> "HRCPipeline":
        """Load model engines, sharing one engine when both model IDs match."""
        force_offline = bool(kwargs.pop("force_offline", False))
        load_in_4bit = bool(kwargs.pop("load_in_4bit", False))
        memory_tau_res = float(kwargs.pop("memory_tau_res", 0.45))
        memory_tau_inject = float(kwargs.pop("memory_tau_inject", 0.78))
        config = HRCConfig(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            **kwargs,
        )
        stage1_engine = _load_engine(
            config.stage1_model,
            device=config.device,
            dtype=config.dtype,
            force_offline=force_offline,
            load_in_4bit=load_in_4bit,
        )
        if config.stage2_mode == "off" or config.stage2_model == config.stage1_model:
            stage2_engine = stage1_engine
        else:
            stage2_engine = _load_engine(
                config.stage2_model or config.stage1_model,
                device=config.device,
                dtype=config.dtype,
                force_offline=force_offline,
                load_in_4bit=load_in_4bit,
            )
        return cls(
            config,
            stage1_engine=stage1_engine,
            stage2_engine=stage2_engine,
            memory_tau_res=memory_tau_res,
            memory_tau_inject=memory_tau_inject,
        )

    def _new_memory(self):
        bench = _bench_module()
        return bench.Layer3ResonantMemory(
            tau_res=self._memory_tau_res,
            hybrid_alpha=0.25,
            tau_inject=self._memory_tau_inject,
        )

    def _score_stage2(self, query: str, answer: str) -> tuple[float, dict[str, Any]]:
        if self._stage2_scorer is not None:
            result = self._stage2_scorer(self.config.stage2_mode, query, answer)
            if isinstance(result, Mapping):
                details = dict(result)
                score = float(details.pop("score"))
                return score, details
            return float(result), {}

        if self.stage2_engine is None:
            raise RuntimeError("Stage-2 model is not loaded")
        bench = _bench_module()
        tokenizer = self.stage2_engine.tokenizer
        model = self.stage2_engine.model
        yes_ids, no_ids = bench.resolve_yes_no_token_ids(tokenizer)
        if self.config.stage2_mode == "dual":
            positive = bench._format_instruct_premise_prompt(
                tokenizer, query, negate=False
            )
            negative = bench._format_instruct_premise_prompt(
                tokenizer, query, negate=True
            )
            d_pos, _, _ = bench._stage2_yes_no_logits(
                model, tokenizer, positive, yes_ids, no_ids
            )
            d_neg, _, _ = bench._stage2_yes_no_logits(
                model, tokenizer, negative, yes_ids, no_ids
            )
            return 0.5 * (d_pos - d_neg), {"d_pos": d_pos, "d_neg": d_neg}
        if self.config.stage2_mode == "direct":
            prompt = bench._format_direct_false_premise_prompt(tokenizer, query)
            log_yes_minus_log_no, _, _ = bench._stage2_yes_no_logits(
                model, tokenizer, prompt, yes_ids, no_ids
            )
            # Higher scores favor a valid premise, matching the cascade rule.
            return -log_yes_minus_log_no, {"log_no_minus_log_yes": -log_yes_minus_log_no}
        raise RuntimeError("Stage 2 is disabled")

    def query(
        self,
        prompt: str,
        *,
        use_memory: bool = True,
        run_stage2: bool = True,
    ) -> HRCResult:
        if self.stage1_engine is None:
            raise RuntimeError("No Stage-1 engine is loaded; use from_pretrained() or inject an engine")
        started = time.perf_counter()
        memory = self.memory if (use_memory and self.config.enable_layer3) else None
        stage1 = self.stage1_engine.infer_with_vm(
            prompt,
            memory=memory,
            v_crit=self.config.vm_threshold,
            max_new_tokens=self.config.max_new_tokens,
            prompt_style=self.config.stage1_prompt_style,
        )
        answer = str(stage1.get("answer", ""))
        raw_answer = str(stage1.get("raw_answer", answer))
        vm_mv = float(stage1["v_m"])
        memory_injected = bool(stage1.get("res_injected", False))
        memory_slot = stage1.get("res_id") if memory_injected else None
        stage2_score: Optional[float] = None
        metadata: dict[str, Any] = {
            "phi_sem": stage1.get("phi_sem"),
            "trajectory_answers": stage1.get("trajectory_answers"),
            "memory_entity_match": stage1.get("res_entity_match"),
            "memory_score": stage1.get("res_sim"),
        }

        if vm_mv > self.config.vm_threshold:
            decision = "HALT_STAGE1"
        elif run_stage2 and self.config.stage2_mode != "off":
            stage2_score, stage2_metadata = self._score_stage2(prompt, answer)
            metadata.update(stage2_metadata)
            decision = (
                "HALT_STAGE2"
                if stage2_score < self.config.stage2_cutoff
                else "ACCEPT"
            )
        else:
            decision = "ACCEPT"

        return HRCResult(
            query=prompt,
            answer=answer,
            raw_answer=raw_answer,
            decision=decision,
            halted=decision != "ACCEPT",
            vm_mv=vm_mv,
            stage2_score=stage2_score,
            stage2_mode=self.config.stage2_mode,
            memory_injected=memory_injected,
            memory_slot=memory_slot,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def batch_query(self, prompts: Sequence[str], **kwargs: Any) -> list[HRCResult]:
        return [self.query(prompt, **kwargs) for prompt in prompts]

    def memorize(self, slot_id: str, entity: str, fact: str, **kwargs: Any) -> MemoryRecord:
        if self.stage1_engine is None:
            raise RuntimeError("A Stage-1 engine is required to embed memory keys")
        if not slot_id or not entity or not fact:
            raise ValueError("slot_id, entity, and fact must be non-empty")
        key_text = str(kwargs.pop("key_text", entity))
        metadata = dict(kwargs.pop("metadata", {}))
        if kwargs:
            metadata.update(kwargs)
        raw_key = self.stage1_engine.embed_texts([key_text])[0]
        self.memory.consolidate(slot_id, raw_key, fact, key_text)
        record = MemoryRecord(slot_id=slot_id, entity=entity, fact=fact, metadata=metadata)
        self._memory_records[slot_id] = record
        # Keep a lightweight raw reference for inspection without affecting the
        # validated memory store's index or retrieval math.
        record.metadata.setdefault("embedding_dim", int(raw_key.numel()))
        return record

    def purge(self, slot_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        purged = bool(self.memory.exact_unlearn(slot_id))
        self._memory_records.pop(slot_id, None)
        purge_ms = (time.perf_counter() - started) * 1000.0
        return {
            "slot_id": slot_id,
            "purged": purged,
            "purge_ms": purge_ms,
            "index_consistent": bool(self.memory.index_is_consistent()),
        }

    def clear_memory(self) -> None:
        self.memory = self._new_memory()
        self._memory_records.clear()

    def calibrate_stage2(
        self,
        valid_scores: Sequence[float],
        trap_scores: Sequence[float],
        max_valid_halts: int = 1,
    ) -> dict[str, Any]:
        if max_valid_halts < 0:
            raise ValueError("max_valid_halts must be non-negative")
        rows = [
            {"is_valid": True, "s2_score": float(score)}
            for score in valid_scores
        ] + [
            {"is_valid": False, "s2_score": float(score)}
            for score in trap_scores
        ]
        bench = _bench_module()
        cutoff, _rule, margin = bench._fit_v2_calibration_cutoff(
            rows, score_key="s2_score", max_valid_halts=max_valid_halts
        )
        self.config.stage2_cutoff = float(cutoff)
        return {
            "cutoff": float(cutoff),
            "margin": float(margin),
            "valid_halts": sum(float(score) < cutoff for score in valid_scores),
            "traps_rejected": sum(float(score) < cutoff for score in trap_scores),
        }
