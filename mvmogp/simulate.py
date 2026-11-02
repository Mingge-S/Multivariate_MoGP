"""Synthetic longitudinal cohort with known progression subtypes.

Used to show that the model runs end to end and recovers planted clusters.
The numbers are illustrative only and are not derived from any real cohort.

Each subtype ``k`` has, for every outcome ``d``, a saturating mean curve that
starts at the anchor value at disease duration 0::

    mu_kd(t) = anchor_d + A_kd * (1 - exp(-r_i * t / tau_kd))

where ``r_i`` is a participant-specific rate multiplier shared across
outcomes. Visits are one year apart, observed with Gaussian noise, plus an
optional additive site offset, and clipped at 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_OUTCOMES = ("sara_total", "mfars_total", "adl_total", "hpt_dom_av", "hpt_ndom_av")


@dataclass
class SimulationSpec:
    """Parameters of the synthetic cohort.

    ``subtypes`` maps a subtype name to ``{outcome: (A, tau)}``: the
    asymptotic increase above the anchor and the time constant (years).
    """

    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    subtypes: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=lambda: {
            "rapid": {
                "sara_total": (38.0, 12.0),
                "mfars_total": (85.0, 12.0),
                "adl_total": (32.0, 14.0),
                "hpt_dom_av": (140.0, 15.0),
                "hpt_ndom_av": (155.0, 15.0),
            },
            "slow": {
                "sara_total": (30.0, 30.0),
                "mfars_total": (70.0, 30.0),
                "adl_total": (26.0, 35.0),
                "hpt_dom_av": (90.0, 35.0),
                "hpt_ndom_av": (100.0, 35.0),
            },
            "upper_limb": {
                "sara_total": (32.0, 20.0),
                "mfars_total": (75.0, 20.0),
                "adl_total": (34.0, 12.0),
                "hpt_dom_av": (200.0, 12.0),
                "hpt_ndom_av": (220.0, 12.0),
            },
        }
    )
    proportions: tuple[float, ...] | None = None  # None = equal
    anchor: tuple[float, ...] | None = None  # raw value at t = 0; None = 0 for every outcome
    noise_sd: dict[str, float] = field(
        default_factory=lambda: {
            "sara_total": 2.0,
            "mfars_total": 4.0,
            "adl_total": 1.8,
            "hpt_dom_av": 8.0,
            "hpt_ndom_av": 10.0,
        }
    )
    rate_sd: float = 0.15  # SD of log rate multiplier per participant
    baseline_range: tuple[float, float] = (0.5, 20.0)  # disease duration at first visit (years)
    visit_probs: tuple[float, float, float] = (0.10, 0.15, 0.75)  # P(1, 2, 3 visits)
    visit_interval: float = 1.0
    n_sites: int = 4
    site_effect_scale: float = 0.75  # site offset SD as a multiple of noise_sd (site 1 has 0)
    clip_at_zero: bool = True

    def anchor_values(self) -> np.ndarray:
        if self.anchor is None:
            return np.zeros(len(self.outcomes))
        return np.asarray(self.anchor, dtype=float)

    def mean_curve(self, subtype: str, t, rate: float = 1.0) -> np.ndarray:
        """Noise-free raw trajectories ``(len(t), n_outcomes)`` for one subtype."""
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        params = np.array([self.subtypes[subtype][o] for o in self.outcomes])
        amplitude, tau = params[:, 0], params[:, 1]
        return self.anchor_values() + amplitude * (1.0 - np.exp(-rate * t / tau))


def simulate_cohort(
    n_participants: int = 150,
    spec: SimulationSpec | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw a synthetic cohort.

    Returns
    -------
    data : long table with ``participant_id, site, visit, disease_duration``
        and one column per outcome (raw units).
    truth : ``participant_id, true_subtype, true_cluster, rate_multiplier, site``.
    """

    spec = spec or SimulationSpec()
    rng = np.random.default_rng(seed)
    names = list(spec.subtypes)
    probs = np.full(len(names), 1.0 / len(names)) if spec.proportions is None else np.asarray(spec.proportions)
    probs = probs / probs.sum()
    noise = np.array([spec.noise_sd[o] for o in spec.outcomes])

    site_offsets = rng.normal(0.0, spec.site_effect_scale, size=(spec.n_sites, len(spec.outcomes))) * noise
    site_offsets[0] = 0.0  # reference site

    rows, truth = [], []
    for i in range(n_participants):
        pid = f"SIM{i + 1:04d}"
        k = int(rng.choice(len(names), p=probs))
        rate = float(np.exp(rng.normal(0.0, spec.rate_sd)))
        site = int(rng.integers(spec.n_sites))
        n_visits = int(rng.choice([1, 2, 3], p=spec.visit_probs))
        t0 = float(rng.uniform(*spec.baseline_range))
        times = t0 + spec.visit_interval * np.arange(n_visits)

        y = spec.mean_curve(names[k], times, rate) + site_offsets[site] + rng.normal(0.0, noise, (n_visits, noise.size))
        if spec.clip_at_zero:
            y = np.clip(y, 0.0, None)

        for v in range(n_visits):
            rows.append(
                {
                    "participant_id": pid,
                    "site": f"Site_{site + 1}",
                    "visit": v + 1,
                    "disease_duration": round(float(times[v]), 2),
                    **{o: float(y[v, d]) for d, o in enumerate(spec.outcomes)},
                }
            )
        truth.append(
            {"participant_id": pid, "true_subtype": names[k], "true_cluster": k, "rate_multiplier": rate,
             "site": f"Site_{site + 1}"}
        )

    return pd.DataFrame(rows), pd.DataFrame(truth)


def true_trajectories(spec: SimulationSpec | None = None, t_min: float = 0.0, t_max: float = 25.0,
                      num_points: int = 200) -> pd.DataFrame:
    """Noise-free subtype curves (rate multiplier 1) in long format, raw units."""
    spec = spec or SimulationSpec()
    t = np.linspace(t_min, t_max, num_points)
    frames = []
    for k, name in enumerate(spec.subtypes):
        curve = spec.mean_curve(name, t)
        for d, outcome in enumerate(spec.outcomes):
            frames.append(pd.DataFrame({"true_cluster": k, "true_subtype": name, "time": t, "outcome": outcome,
                                        "mean": curve[:, d]}))
    return pd.concat(frames, ignore_index=True)
