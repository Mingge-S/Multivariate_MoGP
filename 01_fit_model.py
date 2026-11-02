"""Fit the anchored multivariate MoGP to a longitudinal table and save the results.

The input is a long CSV with one row per visit: a participant id, a time
column (disease duration in years) and one column per outcome in raw units.
Settings come from a JSON config (see configs/) and can be overridden with
the flags below or with any number of --set section.field=value.

Examples:
  python 01_fit_model.py                                   # dummy data, configs/demo.json
  python 01_fit_model.py --kernel matern52 --lengthscale-min 2 --noise-min 0.05
  python 01_fit_model.py --set gp.lengthscale_prior=[3,0.5] --set train.lambda_mono=0
  python 01_fit_model.py --data my_cohort.csv --config configs/manuscript.json --out results/my_cohort
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mvmogp import agreement_with_truth, fit_mixture, save_results, summary_metrics  # noqa: E402
from mvmogp.pipeline import load_run_config, prepare_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "dummy" / "dummy_longitudinal.csv")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "demo.json")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "demo")
    p.add_argument("--truth", type=Path, default=None,
                   help="CSV with participant_id,true_cluster to report recovery (simulated data only)")

    g = p.add_argument_group("common overrides")
    g.add_argument("--n-clusters", type=int)
    g.add_argument("--anchor", choices=["hard", "soft", "none"])
    g.add_argument("--anchor-raw", type=float, nargs="+", help="raw anchor value(s) at t=0")
    g.add_argument("--kernel", choices=["rbf", "matern12", "matern32", "matern52", "rq"])
    g.add_argument("--lengthscale-min", type=float)
    g.add_argument("--lengthscale-init", type=float)
    g.add_argument("--noise-min", type=float, help="lower bound on each noise variance (z-units^2)")
    g.add_argument("--noise-init", type=float, help="initial total noise variance per outcome (z-units^2)")
    g.add_argument("--fix-noise", action="store_true", help="keep noise at --noise-init instead of learning it")
    g.add_argument("--gp-iters", type=int)
    g.add_argument("--em-iters", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--lambda-mono", type=float)
    g.add_argument("--lambda-smooth", type=float)
    g.add_argument("--no-site-correction", action="store_true")
    g.add_argument("--no-unassigned", action="store_true", help="assign every participant to a cluster")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"])
    g.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                   help="override any config field, e.g. gp.task_rank=2 (repeatable)")
    return p.parse_args()


def collect_overrides(args: argparse.Namespace) -> list[str]:
    flag_map = {
        "n_clusters": "mixture.n_clusters",
        "anchor": "gp.anchor",
        "kernel": "gp.kernel",
        "lengthscale_min": "gp.lengthscale_min",
        "lengthscale_init": "gp.lengthscale_init",
        "noise_min": "gp.noise_min",
        "noise_init": "gp.noise_init",
        "gp_iters": "train.n_iter",
        "em_iters": "mixture.max_em_iters",
        "lr": "train.lr",
        "lambda_mono": "train.lambda_mono",
        "lambda_smooth": "train.lambda_smooth",
        "device": "train.device",
    }
    overrides = [f"{key}={json.dumps(getattr(args, flag))}" for flag, key in flag_map.items()
                 if getattr(args, flag) is not None]
    if args.anchor_raw is not None:
        value = args.anchor_raw[0] if len(args.anchor_raw) == 1 else args.anchor_raw
        overrides.append(f"data.anchor_raw={json.dumps(value)}")
    if args.fix_noise:
        overrides.append("gp.learn_noise=false")
    if args.no_site_correction:
        overrides.append("data.site_correction=false")
    if args.no_unassigned:
        overrides.append("mixture.allow_unassigned=false")
    return overrides + list(args.set)


def main() -> None:
    args = parse_args()
    data_cfg, model_cfg = load_run_config(args.config, collect_overrides(args))

    raw = pd.read_csv(args.data)
    prep = prepare_data(raw, data_cfg)
    print(f"Data: {len(prep['data'])} visits from {prep['data'][data_cfg.id_col].nunique()} participants")
    print("Anchor (raw -> z):", dict(zip(data_cfg.outcomes, prep["anchor_z"].round(3))))

    result = fit_mixture(
        prep["times"], prep["values"], prep["patient_ids"],
        anchor_z=prep["anchor_z"], config=model_cfg, outcome_names=prep["z_cols"],
    )

    metrics = summary_metrics(result, prep["times"], prep["values"], prep["patient_ids"],
                              t_min=data_cfg.t_min, t_max=data_cfg.t_max)
    if args.truth is not None:
        truth = pd.read_csv(args.truth)
        agreement = agreement_with_truth(result, dict(zip(truth[data_cfg.id_col], truth["true_cluster"])))
        metrics["recovery"] = {"ari_assigned": agreement["ari"], "n_assigned": agreement["n"],
                               "contingency": agreement["contingency"]}
        print("\nRecovery of planted subtypes (assigned participants)")
        print(f"  adjusted Rand index = {agreement['ari']:.3f}")
        print(agreement["contingency"].to_string())

    written = save_results(
        args.out, result, data=prep["data"], scaler=prep["scaler"], metrics=metrics,
        id_col=data_cfg.id_col, t_min=data_cfg.t_min, t_max=data_cfg.t_max,
        extra_metadata={"data_config": vars(data_cfg), "input_file": args.data.name,
                        "site_offsets": prep["site_offsets"]},
    )

    print("\nCluster sizes (-1 = unassigned):", result.cluster_sizes())
    print(f"EM iterations: {result.n_em_iters} (converged: {result.converged})")
    print(f"Total log-likelihood: {metrics['total_loglik']:.1f}   BIC: {metrics['bic']:.1f}   "
          f"parameters: {metrics['n_parameters']}")
    for k, hp in result.hyperparameters().items():
        print(f"  cluster {k}: lengthscale={hp['lengthscale']:.2f}  outputscale={hp['outputscale']:.3f}  "
              f"slopes={[round(s, 3) for s in hp['slopes']]}")
    print(f"\nSaved {len(written)} files to {args.out}")


if __name__ == "__main__":
    main()
