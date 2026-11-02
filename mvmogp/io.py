"""Save and reload a fitted mixture together with its tables."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig
from .mixture import MixtureResult
from .model import MultitaskProgressionGP, build_likelihood
from .preprocessing import ZScoreScaler

CHECKPOINT_NAME = "model_checkpoint.pth"


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="split")
    if isinstance(value, Path):
        return str(value)
    return value


def save_results(
    output_dir: str | Path,
    result: MixtureResult,
    data: pd.DataFrame | None = None,
    scaler: ZScoreScaler | None = None,
    metrics: dict | None = None,
    id_col: str = "participant_id",
    t_min: float = 0.0,
    t_max: float = 25.0,
    num_points: int = 200,
    extra_metadata: dict | None = None,
) -> list[str]:
    """Write the checkpoint and human-readable tables to ``output_dir``.

    Files: ``model_checkpoint.pth``, ``assignments.csv``, ``diagnostics.csv``,
    ``training_log.csv``, ``trajectories_predictive.csv``,
    ``trajectories_latent.csv`` (both in model units, plus ``*_raw.csv`` when a
    scaler is given), ``hyperparameters.json``, ``metrics.json``,
    ``config.json``, ``metadata.json`` and, if ``data`` is given,
    ``data_with_clusters.csv``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    checkpoint = {
        "format_version": 1,
        "config": result.config.to_dict(),
        "outcome_names": result.outcome_names,
        "anchor_z": None if result.anchor_z is None else np.asarray(result.anchor_z).tolist(),
        "assignments": result.assignments,
        "n_em_iters": result.n_em_iters,
        "converged": result.converged,
        "scaler": scaler.to_dict() if scaler is not None else None,
        "clusters": [
            None
            if m is None
            else {
                "model_state": {k: v.detach().cpu() for k, v in m.state_dict().items()},
                "likelihood_state": {k: v.detach().cpu() for k, v in lik.state_dict().items()},
                "train_x": m.train_inputs[0].detach().cpu(),
                "train_y": m.train_targets.detach().cpu(),
            }
            for m, lik in zip(result.models, result.likelihoods)
        ],
    }
    torch.save(checkpoint, out / CHECKPOINT_NAME)
    written.append(CHECKPOINT_NAME)

    pd.DataFrame(
        sorted(result.assignments.items(), key=lambda kv: str(kv[0])), columns=[id_col, "cluster"]
    ).to_csv(out / "assignments.csv", index=False)
    written.append("assignments.csv")

    result.diagnostics.rename(columns={"participant_id": id_col}).to_csv(out / "diagnostics.csv", index=False)
    result.training_log.to_csv(out / "training_log.csv", index=False)
    written += ["diagnostics.csv", "training_log.csv"]
    if not result.restarts.empty:
        result.restarts.to_csv(out / "restarts.csv", index=False)
        written.append("restarts.csv")

    for band in ("predictive", "latent"):
        traj = result.predict(t_min, t_max, num_points, band=band)
        traj.to_csv(out / f"trajectories_{band}.csv", index=False)
        written.append(f"trajectories_{band}.csv")
        if scaler is not None:
            scaler.trajectories_to_raw(traj).to_csv(out / f"trajectories_{band}_raw.csv", index=False)
            written.append(f"trajectories_{band}_raw.csv")

    if data is not None:
        labelled = data.copy()
        labelled["cluster"] = result.labels_for(labelled[id_col])
        labelled.to_csv(out / "data_with_clusters.csv", index=False)
        written.append("data_with_clusters.csv")

    files = {
        "hyperparameters.json": result.hyperparameters(),
        "metrics.json": metrics or {},
        "config.json": result.config.to_dict(),
        "metadata.json": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outcome_names": result.outcome_names,
            "anchor_z": result.anchor_z,
            "scaler": scaler.to_dict() if scaler is not None else None,
            **(extra_metadata or {}),
        },
    }
    for name, content in files.items():
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(_jsonify(content), f, indent=2)
        written.append(name)
    return written


def _load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=map_location)


def load_results(output_dir: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Rebuild a ``MixtureResult`` (and scaler) from ``save_results`` output.

    Returns a dict with keys ``result``, ``scaler`` and ``data`` (the saved
    ``data_with_clusters.csv`` or ``None``).
    """

    out = Path(output_dir)
    ck = _load_checkpoint(out / CHECKPOINT_NAME, map_location)
    config = ModelConfig.from_dict(ck["config"])
    anchor_z = None if ck["anchor_z"] is None else np.asarray(ck["anchor_z"], dtype=float)

    models, likelihoods = [], []
    for cluster in ck["clusters"]:
        if cluster is None:
            models.append(None)
            likelihoods.append(None)
            continue
        train_x = cluster["train_x"].to(map_location).view(-1, 1)
        train_y = cluster["train_y"].to(map_location)
        num_tasks = int(train_y.shape[-1])
        likelihood = build_likelihood(num_tasks, config.gp).to(map_location)
        model = MultitaskProgressionGP(train_x, train_y, likelihood, num_tasks, config.gp, anchor_z).to(map_location)
        model.load_state_dict(cluster["model_state"])
        likelihood.load_state_dict(cluster["likelihood_state"])
        model.eval()
        likelihood.eval()
        models.append(model)
        likelihoods.append(likelihood)

    diag_path = out / "diagnostics.csv"
    log_path = out / "training_log.csv"
    data_path = out / "data_with_clusters.csv"
    result = MixtureResult(
        assignments=dict(ck["assignments"]),
        models=models,
        likelihoods=likelihoods,
        config=config,
        outcome_names=list(ck["outcome_names"]),
        anchor_z=anchor_z,
        diagnostics=pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame(),
        training_log=pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(),
        n_em_iters=int(ck.get("n_em_iters", 0)),
        converged=bool(ck.get("converged", False)),
    )
    scaler = ZScoreScaler.from_dict(ck["scaler"]) if ck.get("scaler") else None
    data = pd.read_csv(data_path) if data_path.exists() else None
    return {"result": result, "scaler": scaler, "data": data}
