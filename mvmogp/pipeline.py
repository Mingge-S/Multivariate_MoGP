"""Glue used by the numbered scripts: run configuration and data preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import ModelConfig
from .preprocessing import ZScoreScaler, site_correct


@dataclass
class DataConfig:
    """Which columns to model and how to prepare them."""

    outcomes: list[str] = field(
        default_factory=lambda: ["sara_total", "mfars_total", "adl_total", "hpt_dom_av", "hpt_ndom_av"]
    )
    outcome_labels: dict[str, str] = field(default_factory=dict)
    id_col: str = "participant_id"
    time_col: str = "disease_duration"
    anchor_raw: float | list[float] = 0.0  # raw value at t = 0 (scalar = same for every outcome)
    site_correction: bool = False
    site_col: str = "site"
    site_time_col: str = "visit"  # time covariate in the site-correction mixed model
    t_min: float = 0.0
    t_max: float = 25.0


def load_run_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> tuple[DataConfig, ModelConfig]:
    """Read a JSON run configuration and apply ``section.field=value`` overrides.

    Override values are parsed as JSON when possible (``gp.lengthscale_prior=[3,0.5]``,
    ``gp.slope_prior_sd=null``) and kept as strings otherwise (``gp.kernel=matern52``).
    """

    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    raw = {section: dict(values) for section, values in raw.items()}

    for item in overrides:
        key, _, value = item.partition("=")
        section, _, name = key.strip().partition(".")
        if not name or section not in ("data", "gp", "train", "mixture"):
            raise ValueError(f"Override must look like section.field=value with section in data/gp/train/mixture: {item!r}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        raw.setdefault(section, {})[name] = parsed

    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig.from_dict({k: v for k, v in raw.items() if k in ("gp", "train", "mixture")})
    return data_cfg, model_cfg


def prepare_data(df: pd.DataFrame, cfg: DataConfig) -> dict[str, Any]:
    """Site-correct (optional), drop incomplete visits, z-score, convert the anchor.

    Returns a dict with ``data`` (long table incl. ``*_z`` columns), ``scaler``,
    ``anchor_z``, ``times``, ``values``, ``patient_ids``, ``z_cols`` and
    ``site_offsets`` (``None`` without site correction).
    """

    needed = [cfg.id_col, cfg.time_col, *cfg.outcomes]
    if cfg.site_correction:
        needed += [cfg.site_col, cfg.site_time_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Input data is missing columns: {missing}")

    data = df.copy()
    site_offsets = None
    if cfg.site_correction:
        data, site_offsets = site_correct(
            data, cfg.outcomes, site_col=cfg.site_col, time_col=cfg.site_time_col, id_col=cfg.id_col
        )

    n_before = len(data)
    data = data.dropna(subset=[cfg.time_col, *cfg.outcomes]).reset_index(drop=True)
    if len(data) < n_before:
        print(f"Dropped {n_before - len(data)} visits with a missing outcome or time.")
    dup = data.duplicated(subset=[cfg.id_col, cfg.time_col]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (participant, time) rows; each visit must have a unique time.")

    scaler = ZScoreScaler()
    data = scaler.fit_transform(data, cfg.outcomes)
    anchor_z = scaler.anchor_to_z(cfg.anchor_raw)

    return {
        "data": data,
        "scaler": scaler,
        "anchor_z": anchor_z,
        "times": data[cfg.time_col].to_numpy(dtype=float),
        "values": data[scaler.z_cols].to_numpy(dtype=float),
        "patient_ids": data[cfg.id_col].to_numpy(),
        "z_cols": scaler.z_cols,
        "site_offsets": site_offsets,
    }


def outcome_labels_for(cfg: DataConfig, scaler: ZScoreScaler) -> dict[str, str]:
    """Display labels for both raw and z columns."""
    labels = {}
    for raw_col in cfg.outcomes:
        name = cfg.outcome_labels.get(raw_col, raw_col)
        labels[raw_col] = name
        labels[raw_col + scaler.suffix] = name
    return labels
