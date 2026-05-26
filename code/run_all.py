"""Master pipeline runner for Paper 2.

Executes all 7 pipeline steps in order, reports timing and exit status.

Usage:
    python code/run_all.py
    python code/run_all.py --skip-data-prep   # skip steps 1–2 if data already prepared
    python code/run_all.py --from 4           # restart from a specific step number
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # second_paper/
CODE = ROOT / "code"

STEPS = [
    (1, "data_prep/01_wisconsin_download.py",
        "Download + preprocess Wisconsin Diagnostic"),
    (2, "data_prep/02_mimic_extract.py",
        "Extract MIMIC-IV breast cancer cohort"),
    (3, "modeling/03_train_models.py",
        "Train + calibrate 5 classifiers (GridSearchCV)"),
    (4, "modeling/04_evaluate_calibration.py",
        "Evaluate calibration metrics -> T1, T2"),
    (5, "dcgs/05_dcgs_analysis.py",
        "Compute DCGS by subgroup -> T3"),
    (6, "conformal/06_conformal_coverage.py",
        "Conformal prediction subgroup coverage -> T4"),
    (7, "figures/07_generate_figures.py",
        "Generate all reliability diagrams + DCGS heatmaps"),
]


def run_step(step_num, rel_path, description):
    script = CODE / rel_path
    print(f"\n{'='*65}")
    print(f"Step {step_num}: {description}")
    print(f"  Script: {script.relative_to(ROOT)}")
    print("="*65)

    t0  = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode}) in {elapsed:.1f}s")
        return False

    print(f"\n  OK -- {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip steps 1 and 2 (data already prepared)")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    args = parser.parse_args()

    start_step = args.from_step
    if args.skip_data_prep:
        start_step = max(start_step, 3)

    print(f"\nPaper 2 Pipeline -- starting from step {start_step}")
    print(f"Root: {ROOT}")
    t_total = time.time()

    failed_at = None
    for step_num, rel_path, description in STEPS:
        if step_num < start_step:
            print(f"  Step {step_num}: skipped ({description})")
            continue
        ok = run_step(step_num, rel_path, description)
        if not ok:
            failed_at = step_num
            break

    elapsed_total = time.time() - t_total
    print(f"\n{'='*65}")
    if failed_at is not None:
        print(f"Pipeline FAILED at step {failed_at}. "
              f"Total elapsed: {elapsed_total:.1f}s")
        sys.exit(1)
    else:
        print(f"Pipeline COMPLETE in {elapsed_total:.1f}s")
        print(f"\nOutputs:")
        print(f"  Tables   : {ROOT / 'tables'}")
        print(f"  Figures  : {ROOT / 'figures'}")
        print(f"  Models   : {ROOT / 'data' / 'models'}")


if __name__ == "__main__":
    main()
