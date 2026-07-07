"""CLI for hierarchical Ki training (K0–K6) on h24 with loss benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_data import save_h24_paired_arrays
from training.profile import profile_ki_wcet
from training.trainer import benchmark_losses, load_spectrogram_cache, prepare_ki_arrays, train_ki
from utils.classifier_registry import ClassifierRegistry
from utils.labels import KI_REGISTRY, is_deterministic_ki, threshold_hi_for_ki






DEFAULT_H24_DIR = Path("datasets/h24/h24")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
ALL_KI = tuple(KI_REGISTRY.keys())

# Per-Ki losses for stratified splits (avoids stale loss_benchmark_*.json from old val runs).
DEFAULT_KI_LOSS: dict[str, str] = {
    "K0": "weighted_ce",
    "K1": "focal",
    "K2": "weighted_ce",
    "K3": "focal",
    "K4": "weighted_ce",
    "K5": "focal",
    "K6": "weighted_ce",
    "Kdet": "weighted_ce",
}


def resolve_loss(ki_name: str, cli_loss: str | None) -> str:
    if cli_loss:
        return cli_loss
    return DEFAULT_KI_LOSS.get(ki_name, "weighted_ce")


def resolve_threshold(ki_name: str, cli_threshold: float | None) -> float | None:
    if is_deterministic_ki(ki_name):
        return None
    if cli_threshold is not None:
        return cli_threshold
    return threshold_hi_for_ki(ki_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical Ki on h24")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--benchmark-losses", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--ki", choices=[*ALL_KI, "all"], default="K2")
    parser.add_argument(
        "--loss",
        default=None,
        help="Force one loss for all Ki; default uses DEFAULT_KI_LOSS per classifier",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_H24_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--benchmark-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--no-augment", action="store_true", help="Disable SpecAugment on train")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run inline WCET timing after each Ki (default: off; use --profile-wcet after train)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile on CUDA during training",
    )
    parser.add_argument("--profile-wcet", action="store_true", help="Profile WCET only (no training)")
    parser.add_argument(
        "--threshold-hi",
        type=float,
        default=None,
        help="Override H_i for all Ki; default uses paper thresholds (0.95 inter/spec, 0.90 global)",
    )
    parser.add_argument(
        "--export-registry",
        action="store_true",
        help="Rebuild classifier_registry.json from existing checkpoints (no training)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Ki whose metrics already include p_idk (partial run recovery)",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Load existing Ki checkpoint and keep training (resets early-stop counter)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stop after this many epochs without val F1 improvement (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not any(
        [
            args.preprocess,
            args.benchmark_losses,
            args.train,
            args.train_all,
            args.profile_wcet,
            args.export_registry,
        ]
    ):
        print(
            "Use --preprocess, --benchmark-losses, --train, --train-all, "
            "--profile-wcet, and/or --export-registry"
        )
        return

    if args.export_registry:
        registry = ClassifierRegistry.from_checkpoint_dir(
            args.checkpoint_dir,
            default_threshold_hi=args.threshold_hi,
        )
        out = args.checkpoint_dir / "classifier_registry.json"
        registry.save(out)
        print(registry.summary_table())
        print(f"\nWrote {out} (+ .parquet)")
        return

    if args.preprocess:
        print(f"Preprocessing paired h24: {args.data_dir}")
        mic, geo, meta = save_h24_paired_arrays(
            output_dir=args.processed_dir,
            data_dir=args.data_dir,
        )
        print(f"Paired mic {mic.shape}, geo {geo.shape}, metadata rows {len(meta)}")

    targets = list(ALL_KI) if args.ki == "all" or args.train_all else [args.ki]

    if args.profile_wcet:
        report = []
        for ki in targets:
            timing = profile_ki_wcet(
                ki_name=ki,
                processed_dir=args.processed_dir,
                batch_size=1,
            )
            report.append(timing)
            print(
                f"{ki}: avg={timing['avg_ms']:.2f}ms WCET={timing['wcet_ms']:.2f}ms "
                f"p95={timing['p95_ms']:.2f}ms"
            )
        out = args.checkpoint_dir / "wcet_profile.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {out}")
        return

    augment_train = not args.no_augment

    if args.benchmark_losses:
        for ki in targets:
            benchmark_losses(
                ki_name=ki,
                processed_dir=args.processed_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.benchmark_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )

    if args.train or args.train_all:
        summary_path = args.checkpoint_dir / "training_summary.json"
        reg_out = args.checkpoint_dir / "classifier_registry.json"
        wcet_out = args.checkpoint_dir / "wcet_profile.json"

        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = []

        if reg_out.exists():
            registry = ClassifierRegistry.load(reg_out)
            print(f"Resuming registry ({len(registry)} rows) from {reg_out}")
        else:
            registry = ClassifierRegistry()

        wcet_report: list[dict] = []
        if wcet_out.exists():
            wcet_report = json.loads(wcet_out.read_text())

        multi_ki = len(targets) > 1
        cache = load_spectrogram_cache(args.processed_dir) if multi_ki else None
        if multi_ki:
            print(f"Loaded shared spectrogram cache for {len(targets)} Ki")

        for ki in targets:
            metrics_path = args.checkpoint_dir / f"{ki}_metrics.json"
            if (
                args.resume
                and not args.continue_training
                and metrics_path.exists()
            ):
                prior = json.loads(metrics_path.read_text())
                if prior.get("p_idk") is not None:
                    print(f"\n>>> Skip {ki} (already has p_idk={prior['p_idk']:.4f})")
                    continue

            loss_key = resolve_loss(ki, args.loss)
            hi = resolve_threshold(ki, args.threshold_hi)
            mode = "Continue" if args.continue_training else "Full train"
            print(f"\n>>> {mode} {ki} with loss={loss_key}, H_i={hi}, patience={args.patience}")
            spec = KI_REGISTRY[ki]
            arrays = prepare_ki_arrays(spec, args.processed_dir, cache=cache)
            result = train_ki(
                ki_name=ki,
                processed_dir=args.processed_dir,
                checkpoint_dir=args.checkpoint_dir,
                loss_key=loss_key,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                patience=args.patience,
                warmup_epochs=args.warmup_epochs,
                augment_train=augment_train,
                arrays=arrays,
                profile_inference=args.profile,
                use_torch_compile=not args.no_compile,
                threshold_hi=hi,
                registry=registry,
                resume_from_checkpoint=args.continue_training,
            )

            p_success = result.best_metrics.get("accuracy")
            row = {
                "ki": result.ki,
                "loss": result.loss_key,
                "threshold_hi": result.threshold_hi,
                "val_macro_f1": result.best_val_macro_f1,
                "val_loss": result.best_val_loss,
                "params": result.num_params,
                "inference_avg_ms": result.inference_avg_ms,
                "inference_wcet_ms": result.inference_wcet_ms,
                "p_idk": result.p_idk,
                "p_success": p_success,
                "p_correct": p_success,
                "checkpoint": result.checkpoint,
            }
            summary = [s for s in summary if s.get("ki") != ki] + [row]
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"Wrote {summary_path} (through {ki})")

            print(f">>> WCET profile {ki} (batch_size=1)")
            timing = profile_ki_wcet(
                ki_name=ki,
                processed_dir=args.processed_dir,
                batch_size=1,
            )
            wcet_report = [e for e in wcet_report if e.get("ki") != ki] + [timing]
            wcet_out.write_text(json.dumps(wcet_report, indent=2))

            rec = registry.get(ki)
            if rec is not None:
                rec.runtime_ms = timing["avg_ms"]
                rec.wcet_ms = timing["wcet_ms"]
                registry.upsert(rec)
            print(
                f"{ki}: avg={timing['avg_ms']:.2f}ms WCET={timing['wcet_ms']:.2f}ms "
                f"p_idk={result.p_idk:.4f} p_success={p_success:.4f}"
            )

            registry.save(reg_out)
            print(registry.summary_table())
            print(f"Wrote {reg_out} (+ .parquet)")


if __name__ == "__main__":
    main()
