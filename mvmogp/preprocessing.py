"""Data preparation: wide-to-long reshaping, site correction, z-scoring, anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


def wide_to_long(
    df_wide: pd.DataFrame,
    outcomes: Sequence[str],
    n_visits: int = 3,
    id_col: str = "participant_id",
    base_time_col: str = "base_duration",
    visit_interval: float = 1.0,
    require_complete: bool = True,
    visit_suffix: str = "_v{visit}",
) -> pd.DataFrame:
    """Reshape one-row-per-participant data into one row per visit.

    Expects columns such as ``sara_total_v1 ... sara_total_v3`` and a
    baseline time column. The time of visit ``v`` is
    ``base_time + (v - 1) * visit_interval``.

    With ``require_complete=True`` a visit is kept only if every outcome was
    recorded at that visit (as in the manuscript).
    """

    records = []
    for _, row in df_wide.iterrows():
        for v in range(1, n_visits + 1):
            scores = {o: row.get(o + visit_suffix.format(visit=v), np.nan) for o in outcomes}
            if require_complete and any(pd.isna(x) for x in scores.values()):
                continue
            if not require_complete and all(pd.isna(x) for x in scores.values()):
                continue
            records.append(
                {
                    id_col: row[id_col],
                    "visit": v,
                    "disease_duration": row[base_time_col] + (v - 1) * visit_interval,
                    **scores,
                }
            )
    return pd.DataFrame(records, columns=[id_col, "visit", "disease_duration", *outcomes])


def site_correct(
    df_long: pd.DataFrame,
    outcomes: Sequence[str],
    site_col: str = "site",
    time_col: str = "visit",
    id_col: str = "participant_id",
    suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove additive site effects with a linear mixed model.

    For each outcome, fits ``score ~ time + C(site)`` with a random intercept
    per participant and subtracts the estimated site coefficient (the
    reference site is unchanged). Requires ``statsmodels``.

    Returns the corrected data (columns ``outcome + suffix``) and a table of
    the subtracted site offsets.
    """

    import statsmodels.formula.api as smf

    out = df_long.copy()
    offsets = []
    for outcome in outcomes:
        use = out.dropna(subset=[outcome, site_col, time_col, id_col])
        data = pd.DataFrame(
            {"score": use[outcome], "time": use[time_col], "site": use[site_col].astype(str), "pid": use[id_col]}
        )
        fit = smf.mixedlm("score ~ time + C(site)", data, groups=data["pid"]).fit()
        coefs = {
            name.replace("C(site)[T.", "").rstrip("]"): float(value)
            for name, value in fit.params.items()
            if name.startswith("C(site)")
        }
        site_effect = out[site_col].astype(str).map(coefs).fillna(0.0)
        out[outcome + suffix] = out[outcome] - site_effect
        offsets += [{"outcome": outcome, "site": s, "offset": c} for s, c in coefs.items()]
    return out, pd.DataFrame(offsets, columns=["outcome", "site", "offset"])


@dataclass
class ZScoreScaler:
    """Per-outcome z-scoring with the mapping back to raw units.

    ``fit`` stores each raw column's mean and SD (ddof=1). ``transform`` adds
    the ``<col><suffix>`` z-score columns. ``anchor_to_z`` converts raw anchor
    values (e.g. severity 0 at onset) to the same z-scale.
    """

    raw_cols: list[str] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    suffix: str = "_z"

    @property
    def z_cols(self) -> list[str]:
        return [c + self.suffix for c in self.raw_cols]

    def fit(self, df: pd.DataFrame, raw_cols: Sequence[str]) -> "ZScoreScaler":
        self.raw_cols = list(raw_cols)
        self.means = {c: float(df[c].mean()) for c in self.raw_cols}
        self.stds = {c: float(df[c].std(ddof=1)) for c in self.raw_cols}
        bad = [c for c, s in self.stds.items() if not np.isfinite(s) or s <= 0]
        if bad:
            raise ValueError(f"Columns with zero or undefined SD cannot be z-scored: {bad}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.raw_cols:
            out[c + self.suffix] = (out[c] - self.means[c]) / self.stds[c]
        return out

    def fit_transform(self, df: pd.DataFrame, raw_cols: Sequence[str]) -> pd.DataFrame:
        return self.fit(df, raw_cols).transform(df)

    def anchor_to_z(self, anchor_raw: float | Sequence[float] = 0.0) -> np.ndarray:
        """Convert raw anchor value(s) to z-units; a scalar applies to every outcome."""
        anchor = np.asarray(anchor_raw, dtype=float)
        if anchor.ndim == 0:
            anchor = np.full(len(self.raw_cols), float(anchor))
        if anchor.size != len(self.raw_cols):
            raise ValueError(f"Expected {len(self.raw_cols)} anchor values, got {anchor.size}.")
        means = np.array([self.means[c] for c in self.raw_cols])
        stds = np.array([self.stds[c] for c in self.raw_cols])
        return (anchor - means) / stds

    def raw_col_for(self, outcome: str) -> str:
        """Raw column for a z-column (or the column itself if already raw)."""
        if outcome in self.raw_cols:
            return outcome
        if outcome.endswith(self.suffix) and outcome[: -len(self.suffix)] in self.raw_cols:
            return outcome[: -len(self.suffix)]
        raise KeyError(f"{outcome!r} is not a scaled outcome.")

    def to_raw(self, outcome: str, values, is_sd: bool = False) -> np.ndarray:
        """Map z-values (or z-SDs with ``is_sd=True``) of ``outcome`` back to raw units."""
        col = self.raw_col_for(outcome)
        values = np.asarray(values, dtype=float)
        return values * self.stds[col] if is_sd else values * self.stds[col] + self.means[col]

    def trajectories_to_raw(self, traj: pd.DataFrame) -> pd.DataFrame:
        """Convert a ``MixtureResult.predict`` table to raw units."""
        out = traj.copy()
        raw_names = out["outcome"].map(self.raw_col_for)
        mu = raw_names.map(self.means).to_numpy(dtype=float)
        sd = raw_names.map(self.stds).to_numpy(dtype=float)
        for col in ("mean", "lower", "upper"):
            out[col] = out[col].to_numpy(dtype=float) * sd + mu
        out["sd"] = out["sd"].to_numpy(dtype=float) * sd
        out["outcome"] = raw_names
        return out

    def to_dict(self) -> dict:
        return {"raw_cols": self.raw_cols, "means": self.means, "stds": self.stds, "suffix": self.suffix}

    @classmethod
    def from_dict(cls, data: dict) -> "ZScoreScaler":
        return cls(**data)
