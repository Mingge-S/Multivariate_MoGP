"""Generate a synthetic longitudinal cohort with planted progression subtypes.

Writes to data/dummy/ (by default):
  dummy_longitudinal.csv  one row per visit: participant_id, site, visit,
                          disease_duration and five outcome columns (raw units)
  dummy_truth.csv         the planted subtype of each participant
  dummy_true_curves.csv   noise-free subtype curves, for checking recovery

Example:
  python data/generate_dummy_data.py --n-participants 150 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvmogp.simulate import SimulationSpec, simulate_cohort, true_trajectories  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-participants", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--site-effect-scale", type=float, default=0.75,
                        help="site offset SD as a multiple of each outcome's noise SD (0 = no site effects)")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="multiply every outcome's noise SD")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "dummy")
    args = parser.parse_args()

    spec = SimulationSpec(n_sites=args.n_sites, site_effect_scale=args.site_effect_scale)
    spec.noise_sd = {k: v * args.noise_scale for k, v in spec.noise_sd.items()}

    data, truth = simulate_cohort(args.n_participants, spec, seed=args.seed)
    curves = true_trajectories(spec)

    args.out.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out / "dummy_longitudinal.csv", index=False)
    truth.to_csv(args.out / "dummy_truth.csv", index=False)
    curves.to_csv(args.out / "dummy_true_curves.csv", index=False)

    print(f"Participants: {truth.shape[0]}   visits: {data.shape[0]}")
    print("Planted subtypes:", truth["true_subtype"].value_counts().to_dict())
    print("Visits per participant:", data.groupby("participant_id").size().value_counts().sort_index().to_dict())
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
