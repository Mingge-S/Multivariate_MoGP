"""Compare numbers of clusters K with restarts and save a diagnostics table.

For each K the mixture is fitted from several KMeans initialisations; the
restart with the highest total log-likelihood is reported together with BIC,
stability across restarts (mean pairwise ARI), time coverage and roughness.

Examples:
  python 03_select_k.py                                  # dummy data, K = 2..5, 3 restarts
  python 03_select_k.py --k 2 3 4 5 6 --restarts 5 --config configs/manuscript.json --data my_cohort.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mvmogp import select_k  # noqa: E402
from mvmogp.pipeline import load_run_config, prepare_data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "dummy" / "dummy_longitudinal.csv")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "demo.json")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "k_selection.csv")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    args = p.parse_args()

    data_cfg, model_cfg = load_run_config(args.config, ["mixture.verbose=false", *args.set])
    prep = prepare_data(pd.read_csv(args.data), data_cfg)

    table, _ = select_k(
        prep["times"], prep["values"], prep["patient_ids"], prep["anchor_z"],
        k_values=args.k, n_restarts=args.restarts, config=model_cfg, outcome_names=prep["z_cols"],
        t_min=data_cfg.t_min, t_max=data_cfg.t_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()
