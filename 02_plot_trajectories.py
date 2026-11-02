"""Plot cluster progression trajectories from a saved fit.

Produces two figures in <results>/figures/:
  progression_trajectories.<fmt>  each cluster's mean trajectory +/- band, one panel per outcome
  overlay_trajectories.<fmt>      participants' observed trajectories with the cluster GP overlaid

Examples:
  python 02_plot_trajectories.py                              # results/demo
  python 02_plot_trajectories.py --band latent --scale raw
  python 02_plot_trajectories.py --results results/my_cohort --t-max 18 --format pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mvmogp import load_results, plot_overlay_trajectories, plot_progression_trajectories  # noqa: E402
from mvmogp.pipeline import DataConfig, outcome_labels_for  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "demo")
    p.add_argument("--band", choices=["predictive", "latent"], default="predictive",
                   help="predictive: where a new visit is expected; latent: uncertainty of the mean curve")
    p.add_argument("--n-sd", type=float, default=2.0, help="band half-width in SDs")
    p.add_argument("--no-band", action="store_true")
    p.add_argument("--scale", choices=["z", "raw"], default="z")
    p.add_argument("--t-min", type=float, default=None)
    p.add_argument("--t-max", type=float, default=None)
    p.add_argument("--format", default="png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    loaded = load_results(args.results)
    result, scaler, data = loaded["result"], loaded["scaler"], loaded["data"]

    with open(args.results / "metadata.json", "r", encoding="utf-8") as f:
        data_cfg = DataConfig(**json.load(f).get("data_config", {}))
    labels = outcome_labels_for(data_cfg, scaler) if scaler is not None else data_cfg.outcome_labels
    t_min = data_cfg.t_min if args.t_min is None else args.t_min
    t_max = data_cfg.t_max if args.t_max is None else args.t_max

    common = dict(outcome_labels=labels, t_min=t_min, t_max=t_max, band=args.band, n_sd=args.n_sd,
                  show_band=not args.no_band, scale=args.scale, scaler=scaler, dpi=args.dpi)
    fig_dir = args.results / "figures"

    path = fig_dir / f"progression_trajectories.{args.format}"
    plot_progression_trajectories(result, save_path=path, **common)
    print("Saved", path)

    if data is None:
        print("No data_with_clusters.csv in results; skipping the overlay figure.")
        return
    path = fig_dir / f"overlay_trajectories.{args.format}"
    plot_overlay_trajectories(result, data, id_col=data_cfg.id_col, time_col=data_cfg.time_col,
                              save_path=path, **common)
    print("Saved", path)


if __name__ == "__main__":
    main()
