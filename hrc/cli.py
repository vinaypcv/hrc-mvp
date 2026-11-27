"""Command-line interface for the installable HRC pipeline."""

from __future__ import annotations

import argparse
import json

from .pipeline import DEFAULT_MODEL, HRCPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrc",
        description="Run the HRC V_m, premise-audit, and guarded-memory cascade.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Stage-1 and default Stage-2 model ID")
    parser.add_argument("--query", help="Question or prompt to evaluate")
    parser.add_argument("--vm-threshold", type=float, default=-66.0)
    parser.add_argument("--stage2-mode", choices=("dual", "direct", "off"), default="dual")
    parser.add_argument("--stage2-cutoff", type=float, default=0.0)
    parser.add_argument(
        "--memorize",
        action="append",
        metavar="SLOT|ENTITY|FACT",
        help="Seed one entity-guarded fact before querying; may be repeated",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.query:
        parser.error("--query is required (you may also seed memory with --memorize)")

    pipeline = HRCPipeline.from_pretrained(
        stage1_model=args.model,
        vm_threshold=args.vm_threshold,
        stage2_mode=args.stage2_mode,
        stage2_cutoff=args.stage2_cutoff,
        device=args.device,
        dtype=args.dtype,
    )
    for item in args.memorize or []:
        parts = item.split("|", 2)
        if len(parts) != 3 or not all(part.strip() for part in parts):
            parser.error("--memorize must use SLOT|ENTITY|FACT")
        slot, entity, fact = (part.strip() for part in parts)
        pipeline.memorize(slot, entity, fact)

    result = pipeline.query(args.query)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        print(f"Decision: {result.decision}")
        print(f"Answer: {result.answer}")
        print(f"V_m: {result.vm_mv:.3f} mV")
        if result.stage2_score is not None:
            print(f"Stage-2 ({result.stage2_mode}) score: {result.stage2_score:.6f}")
        print(f"Memory injected: {result.memory_injected}")
        print(f"Latency: {result.latency_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
