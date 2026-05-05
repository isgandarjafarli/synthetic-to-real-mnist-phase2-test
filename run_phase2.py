"""Command-line runner for Phase 2 experiments.

Examples:
    python run_phase2.py --mode smoke
    python run_phase2.py --mode full --update-readme
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_to_real_mnist.config import TrainConfig
from synthetic_to_real_mnist.experiments import run_phase2_suite
from synthetic_to_real_mnist.reporting import update_readme_results, write_phase2_report


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 2 synthetic-to-real MNIST experiments")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full", help="smoke is quick; full is the final experiment suite")
    parser.add_argument("--output-dir", default="outputs", help="where plots, histories, models, and result tables are saved")
    parser.add_argument("--data-root", default="./data", help="where MNIST is downloaded/cached")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-save-models", action="store_true", help="do not write model checkpoints")
    parser.add_argument("--update-readme", action="store_true", help="replace the README results section with actual results")
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    df, models, detailed = run_phase2_suite(
        mode=args.mode,
        output_dir=args.output_dir,
        data_root=args.data_root,
        base_train_cfg=train_cfg,
        save_model=not args.no_save_models,
    )
    output_dir = Path(args.output_dir)
    write_phase2_report(df, output_dir / "phase2_report.md")
    if args.update_readme:
        update_readme_results(REPO_ROOT / "README.md", df)
    print("\nDone. Results saved to:")
    print(f"- {output_dir / 'phase2_results.csv'}")
    print(f"- {output_dir / 'phase2_report.md'}")
    print(f"- {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
