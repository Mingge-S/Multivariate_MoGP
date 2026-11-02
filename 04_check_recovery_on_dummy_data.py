"""Check that a fit on the dummy data recovers the planted subtypes.

Matches each fitted cluster to the planted subtype most of its members came
from, reports the adjusted Rand index, and draws the fitted mean trajectories
(raw units, latent band) with the true noise-free subtype curves dashed on top.

Run after data/generate_dummy_data.py and 01_fit_model.py:
  python 04_check_recovery_on_dummy_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mvmogp import agreement_with_truth, load_results, plot_progression_trajectories  # noqa: E402
from mvmogp.plotting import CLUSTER_COLORS, INK  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "demo")
    p.add_argument("--dummy-dir", type=Path, default=REPO_ROOT / "data" / "dummy")
    args = p.parse_args()

    loaded = load_results(args.results)
    result, scaler = loaded["result"], loaded["scaler"]
    truth = pd.read_csv(args.dummy_dir / "dummy_truth.csv")
    curves = pd.read_csv(args.dummy_dir / "dummy_true_curves.csv")

    agreement = agreement_with_truth(result, dict(zip(truth["participant_id"], truth["true_cluster"])))
    table = agreement["contingency"]
    print(f"Adjusted Rand index (assigned participants): {agreement['ari']:.3f}")
    print(table.to_string())

    # Fitted cluster -> planted subtype contributing most members.
    match = {int(k): int(table[k].idxmax()) for k in table.columns if int(k) in result.fitted_clusters}
    names = truth.drop_duplicates("true_cluster").set_index("true_cluster")["true_subtype"].to_dict()
    labels = {k: f"Cluster {i + 1} ~ {names[match[k]]}" for i, k in enumerate(result.fitted_clusters) if k in match}

    fig, axes = plot_progression_trajectories(
        result, scale="raw", scaler=scaler, band="latent", cluster_labels=labels,
        outcome_labels={o: o for o in scaler.raw_cols},
    )
    for ax, outcome in zip(axes, result.outcome_names):
        raw_col = scaler.raw_col_for(outcome)
        for i, k in enumerate(result.fitted_clusters):
            if k not in match:
                continue
            true_curve = curves[(curves["true_cluster"] == match[k]) & (curves["outcome"] == raw_col)]
            ax.plot(true_curve["time"], true_curve["mean"], linestyle="--", linewidth=1.2,
                    color=CLUSTER_COLORS[i], alpha=0.9)
    fig.text(0.005, 0.005, "solid: fitted mean ± 2 SD (latent)   dashed: true subtype curve", fontsize=8, color=INK)

    out = args.results / "figures" / "recovery_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("Saved", out)


if __name__ == "__main__":
    main()
