"""Finite mixture of multitask GPs fitted by hard-assignment EM.

Each participant belongs to exactly one cluster (or is left unassigned).
The algorithm alternates:

1. **M-step** - fit one multitask GP per cluster on the pooled observations
   of its current members (``model.fit_cluster_gp``).
2. **E-step** - move every participant to the cluster whose GP gives the
   highest joint log predictive density for their whole trajectory.
3. **Trimming** - optionally mark the worst-fitting participants of each
   cluster as unassigned, and dissolve clusters that became too small.

It stops when no assignment changes (or after ``max_em_iters``) and then
refits every cluster once more on its final members.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .model import fit_cluster_gp, model_summary, patient_loglik, predict_trajectory


@dataclass
class MixtureResult:
    """Fitted mixture: assignments, one GP per cluster, and diagnostics."""

    assignments: dict[Any, int]
    models: list
    likelihoods: list
    config: ModelConfig
    outcome_names: list[str]
    anchor_z: np.ndarray | None = None
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    training_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_em_iters: int = 0
    converged: bool = False
    restarts: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per restart when n_restarts > 1

    @property
    def unassigned_label(self) -> int:
        return self.config.mixture.unassigned_label

    @property
    def fitted_clusters(self) -> list[int]:
        """Indices of clusters that have a fitted GP."""
        return [k for k, m in enumerate(self.models) if m is not None]

    def cluster_sizes(self) -> dict[int, int]:
        sizes = pd.Series(list(self.assignments.values())).value_counts().sort_index()
        return {int(k): int(v) for k, v in sizes.items()}

    def labels_for(self, participant_ids) -> np.ndarray:
        """Cluster label for each id (unassigned_label for unknown ids)."""
        return np.array([self.assignments.get(pid, self.unassigned_label) for pid in participant_ids], dtype=int)

    def hyperparameters(self) -> dict[int, dict[str, Any]]:
        return {k: model_summary(self.models[k], self.likelihoods[k]) for k in self.fitted_clusters}

    def predict(
        self,
        t_min: float = 0.0,
        t_max: float = 25.0,
        num_points: int = 200,
        band: str = "predictive",
        n_sd: float = 2.0,
    ) -> pd.DataFrame:
        """Cluster trajectories on a regular time grid.

        ``band="predictive"`` includes observation noise (where a new visit is
        expected to fall); ``band="latent"`` is the uncertainty of the mean
        trajectory itself. ``lower``/``upper`` are ``mean -/+ n_sd * sd``.
        """

        if band not in ("predictive", "latent"):
            raise ValueError("band must be 'predictive' or 'latent'.")
        t_grid = np.linspace(t_min, t_max, num_points)
        frames = []
        for k in self.fitted_clusters:
            mean, sd = predict_trajectory(
                self.models[k], self.likelihoods[k], t_grid, include_noise=(band == "predictive")
            )
            if mean is None:
                continue
            for d, outcome in enumerate(self.outcome_names):
                frames.append(
                    pd.DataFrame(
                        {
                            "cluster": k,
                            "time": t_grid,
                            "outcome": outcome,
                            "mean": mean[:, d],
                            "sd": sd[:, d],
                            "lower": mean[:, d] - n_sd * sd[:, d],
                            "upper": mean[:, d] + n_sd * sd[:, d],
                        }
                    )
                )
        columns = ["cluster", "time", "outcome", "mean", "sd", "lower", "upper"]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def patient_features(times, values, patient_ids) -> tuple[list, np.ndarray]:
    """Per-participant summary features used to initialise clusters.

    For ``D`` outcomes the features are: first observed time, the first
    observed value of each outcome, and each outcome's slope between the
    first and last visit (0 for single-visit participants). Shape ``(P, 1 + 2D)``.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    patient_ids = np.asarray(patient_ids)
    unique_patients = sorted(set(patient_ids.tolist()))

    feats = []
    for pid in unique_patients:
        idx = np.where(patient_ids == pid)[0]
        order = np.argsort(times[idx])
        t_i = times[idx][order]
        y_i = values[idx][order]
        dt = float(t_i[-1] - t_i[0])
        slopes = np.zeros(values.shape[1]) if dt <= 1e-6 else (y_i[-1] - y_i[0]) / dt
        feats.append(np.concatenate([[t_i[0]], y_i[0], slopes]))

    X = np.vstack(feats)
    col_means = np.nanmean(X, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = col_means[nan_cols]
    return unique_patients, X


def initialize_kmeans(times, values, patient_ids, n_clusters: int, random_state: int = 0) -> dict[Any, int]:
    """KMeans on standardised ``patient_features``; returns ``{id: cluster}``."""

    unique_patients, X = patient_features(times, values, patient_ids)
    labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(
        StandardScaler().fit_transform(X)
    )
    return {pid: int(lbl) for pid, lbl in zip(unique_patients, labels)}


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


def select_poor_fit(diag: pd.DataFrame, cfg, n_patients: int) -> set:
    """Worst-fitting participants to mark as unassigned.

    Within each cluster of at least ``trim_min_cluster_size`` members, up to
    ``trim_frac_per_cluster`` of members (at least one) whose mean absolute
    standardised residual is at least ``trim_abs_z_threshold`` are selected,
    worst first. The total is capped at ``max_unassigned_frac`` of all
    participants.
    """

    order = dict(by=["mean_abs_z", "best_ll", "ll_margin"], ascending=[False, True, True])
    selected = set()
    for _, group in diag.groupby("best_cluster"):
        if len(group) < cfg.trim_min_cluster_size:
            continue
        n_trim = max(1, int(np.floor(cfg.trim_frac_per_cluster * len(group))))
        worst = group.sort_values(**order)
        worst = worst[worst["mean_abs_z"] >= cfg.trim_abs_z_threshold].head(n_trim)
        selected.update(worst["participant_id"].tolist())

    cap = int(np.floor(cfg.max_unassigned_frac * n_patients))
    if len(selected) > cap:
        worst = diag[diag["participant_id"].isin(selected)].sort_values(**order)
        selected = set(worst.head(cap)["participant_id"].tolist())
    return selected


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------


def _cluster_tensors(times, values, patient_to_idx, members):
    idx = [i for pid in members for i in patient_to_idx[pid]]
    x = torch.tensor(times[idx], dtype=torch.float32).view(-1, 1)
    y = torch.tensor(values[idx], dtype=torch.float32)
    return x, y


def assigned_loglik(result: MixtureResult, times, values, patient_ids) -> float:
    """Sum of log predictive densities of assigned participants under their cluster."""

    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    patient_ids = np.asarray(patient_ids)
    total = 0.0
    for pid in np.unique(patient_ids):
        k = result.assignments.get(pid, result.unassigned_label)
        if 0 <= k < len(result.models) and result.models[k] is not None:
            idx = patient_ids == pid
            total += patient_loglik(result.models[k], result.likelihoods[k], times[idx], values[idx])[0]
    return float(total)


def fit_mixture(
    times,
    values,
    patient_ids,
    anchor_z=None,
    config: ModelConfig | None = None,
    outcome_names: list[str] | None = None,
    initial_assignments: dict[Any, int] | None = None,
) -> MixtureResult:
    """Cluster participants into a finite mixture of multitask GPs.

    Parameters
    ----------
    times : (N,) observation times (one row per visit).
    values : (N, D) outcomes for each visit, usually z-scored. No missing values.
    patient_ids : (N,) participant identifier of each visit.
    anchor_z : (D,) anchor in the units of ``values``; required unless
        ``config.gp.anchor == "none"``.
    config : model, training and EM settings (defaults reproduce the manuscript).
    outcome_names : names of the D outcome columns, used in outputs.
    initial_assignments : optional ``{id: cluster}`` to skip KMeans initialisation.

    With ``config.mixture.n_restarts > 1`` the EM is run from that many KMeans
    seeds and the run with the highest total log-likelihood of assigned
    participants is returned (its ``restarts`` table lists every run).
    """

    config = config or ModelConfig()
    n_restarts = config.mixture.n_restarts
    if n_restarts <= 1 or initial_assignments is not None:
        return _fit_mixture_once(times, values, patient_ids, anchor_z, config, outcome_names, initial_assignments)

    runs, rows = [], []
    for r in range(n_restarts):
        cfg = ModelConfig.from_dict(config.to_dict())
        cfg.mixture.init_random_state = config.mixture.init_random_state + r
        if config.mixture.verbose:
            print(f"\n##### Restart {r + 1}/{n_restarts} (KMeans seed {cfg.mixture.init_random_state}) #####")
        res = _fit_mixture_once(times, values, patient_ids, anchor_z, cfg, outcome_names)
        ll = assigned_loglik(res, times, values, patient_ids)
        runs.append(res)
        rows.append({"restart": r, "init_random_state": cfg.mixture.init_random_state, "total_loglik": ll,
                     "n_em_iters": res.n_em_iters, "converged": res.converged,
                     "n_unassigned": sum(v == res.unassigned_label for v in res.assignments.values()),
                     "cluster_sizes": res.cluster_sizes()})
    table = pd.DataFrame(rows)
    best = runs[int(table["total_loglik"].idxmax())]
    best.restarts = table
    if config.mixture.verbose:
        print("\nRestarts:\n" + table.drop(columns="cluster_sizes").to_string(index=False))
        print(f"Kept KMeans seed {best.config.mixture.init_random_state}.")
    return best


def _fit_mixture_once(
    times,
    values,
    patient_ids,
    anchor_z=None,
    config: ModelConfig | None = None,
    outcome_names: list[str] | None = None,
    initial_assignments: dict[Any, int] | None = None,
) -> MixtureResult:
    """One EM run from one initialisation (see ``fit_mixture``)."""

    config = config or ModelConfig()
    gp_cfg, train_cfg, mix_cfg = config.gp, config.train, config.mixture
    verbose = mix_cfg.verbose

    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    patient_ids = np.asarray(patient_ids)
    if values.ndim != 2 or not (times.shape[0] == values.shape[0] == patient_ids.shape[0]):
        raise ValueError("times (N,), values (N, D) and patient_ids (N,) must have matching lengths.")
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        raise ValueError("times and values must not contain NaN or inf; drop incomplete visits first.")

    num_tasks = values.shape[1]
    outcome_names = list(outcome_names) if outcome_names is not None else [f"y{d}" for d in range(num_tasks)]
    if len(outcome_names) != num_tasks:
        raise ValueError("outcome_names must have one name per column of values.")
    if anchor_z is not None:
        anchor_z = np.asarray(anchor_z, dtype=float).reshape(-1)
        if anchor_z.size != num_tasks:
            raise ValueError(f"anchor_z has {anchor_z.size} values but values has {num_tasks} columns.")
    elif gp_cfg.anchor != "none":
        raise ValueError(f"anchor={gp_cfg.anchor!r} requires anchor_z.")

    n_clusters = mix_cfg.n_clusters
    unassigned = mix_cfg.unassigned_label
    trainable_min_size = max(5, num_tasks + 1)

    patient_to_idx: dict[Any, list[int]] = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        patient_to_idx[pid].append(i)
    unique_patients = sorted(patient_to_idx)
    n_patients = len(unique_patients)

    if verbose:
        print(f"Fitting multivariate MoGP: K={n_clusters}, {n_patients} participants, {len(times)} visits, "
              f"{num_tasks} outcomes, anchor={gp_cfg.anchor}, kernel={gp_cfg.kernel}.")

    if initial_assignments is None:
        assignments = initialize_kmeans(times, values, patient_ids, n_clusters, mix_cfg.init_random_state)
    else:
        assignments = {pid: int(initial_assignments[pid]) for pid in unique_patients}

    models: list = [None] * n_clusters
    likelihoods: list = [None] * n_clusters
    log_rows: list[dict[str, Any]] = []
    diag = pd.DataFrame()
    converged = False
    em_iter = 0

    def members_of(k):
        return [pid for pid in unique_patients if assignments[pid] == k]

    for em_iter in range(1, mix_cfg.max_em_iters + 1):
        if verbose:
            print(f"\n=== EM iteration {em_iter}/{mix_cfg.max_em_iters} ===")
        prev_assignments = dict(assignments)

        # ---- M-step ------------------------------------------------------
        for k in range(n_clusters):
            members = members_of(k)
            if not members:
                models[k] = likelihoods[k] = None
                if verbose:
                    print(f"  [cluster {k}] empty")
                continue
            if len(members) < trainable_min_size:
                if verbose:
                    print(f"  [cluster {k}] only {len(members)} participants; keeping previous model")
                continue
            x, y = _cluster_tensors(times, values, patient_to_idx, members)
            if verbose:
                print(f"  [cluster {k}] fitting on {len(members)} participants, {x.shape[0]} visits")
            models[k], likelihoods[k], log_k = fit_cluster_gp(
                x, y, gp_cfg, train_cfg, anchor_z, verbose=(verbose and k == 0)
            )
            log_rows += [{**row, "stage": "em", "em_iter": em_iter, "cluster": k} for row in log_k]

        # ---- E-step ------------------------------------------------------
        diag_rows = []
        for pid in unique_patients:
            idx = patient_to_idx[pid]
            scores = [
                patient_loglik(models[k], likelihoods[k], times[idx], values[idx])
                if models[k] is not None
                else (-1e9, float("inf"))
                for k in range(n_clusters)
            ]
            lls = np.array([s[0] for s in scores])
            best = int(np.argmax(lls))
            second = float(np.partition(lls, -2)[-2]) if n_clusters >= 2 else float("nan")
            diag_rows.append(
                {
                    "participant_id": pid,
                    "best_cluster": best,
                    "best_ll": float(lls[best]),
                    "second_ll": second,
                    "ll_margin": float(lls[best] - second),
                    "mean_abs_z": scores[best][1],
                    **{f"ll_cluster_{k}": float(lls[k]) for k in range(n_clusters)},
                }
            )
        diag = pd.DataFrame(diag_rows)
        assignments = dict(zip(diag["participant_id"], diag["best_cluster"].astype(int)))

        # ---- Trimming ----------------------------------------------------
        poor_fit = select_poor_fit(diag, mix_cfg, n_patients) if mix_cfg.allow_unassigned else set()
        for pid in poor_fit:
            assignments[pid] = unassigned
        diag["trimmed"] = diag["participant_id"].isin(poor_fit)

        for k in range(n_clusters):
            members = members_of(k)
            if 0 < len(members) < mix_cfg.min_cluster_size:
                if mix_cfg.allow_unassigned:
                    if verbose:
                        print(f"  [cluster {k}] {len(members)} < min_cluster_size; members set to unassigned")
                    for pid in members:
                        assignments[pid] = unassigned
                else:
                    if verbose:
                        print(f"  [cluster {k}] {len(members)} < min_cluster_size; members moved to next best")
                    others = [f"ll_cluster_{kk}" for kk in range(n_clusters) if kk != k and models[kk] is not None]
                    for pid in members:
                        if not others:
                            assignments[pid] = unassigned
                            continue
                        row = diag.loc[diag["participant_id"] == pid, others].iloc[0]
                        assignments[pid] = int(row.idxmax().rsplit("_", 1)[1])

        diag["final_assignment"] = diag["participant_id"].map(assignments)
        changes = sum(assignments[pid] != prev_assignments[pid] for pid in unique_patients)

        if verbose:
            sizes = pd.Series(list(assignments.values())).value_counts().sort_index().to_dict()
            print(f"  unassigned: {sum(v == unassigned for v in assignments.values())}  sizes: {sizes}")
            print(f"  assignment changes: {changes}/{n_patients}")
        if changes == 0:
            converged = True
            if verbose:
                print("  converged.")
            break

    # ---- Final refit on the final members --------------------------------
    if verbose:
        print("\n=== Final refit ===")
    final_models: list = [None] * n_clusters
    final_likelihoods: list = [None] * n_clusters
    for k in range(n_clusters):
        members = members_of(k)
        if len(members) < trainable_min_size:
            if verbose and members:
                print(f"  [cluster {k}] {len(members)} participants < {trainable_min_size}; no model")
            continue
        x, y = _cluster_tensors(times, values, patient_to_idx, members)
        if verbose:
            print(f"  [cluster {k}] {len(members)} participants, {x.shape[0]} visits")
        final_models[k], final_likelihoods[k], log_k = fit_cluster_gp(x, y, gp_cfg, train_cfg, anchor_z)
        log_rows += [{**row, "stage": "final", "em_iter": None, "cluster": k} for row in log_k]

    return MixtureResult(
        assignments={pid: int(c) for pid, c in assignments.items()},
        models=final_models,
        likelihoods=final_likelihoods,
        config=config,
        outcome_names=outcome_names,
        anchor_z=anchor_z,
        diagnostics=diag,
        training_log=pd.DataFrame(log_rows),
        n_em_iters=em_iter,
        converged=converged,
    )
