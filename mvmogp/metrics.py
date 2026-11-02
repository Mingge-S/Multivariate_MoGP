"""Fit diagnostics and model selection for a fitted mixture."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import ModelConfig
from .mixture import MixtureResult, fit_mixture
from .model import patient_loglik, predict_trajectory


def _group_rows(patient_ids) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, pid in enumerate(np.asarray(patient_ids)):
        groups[pid].append(i)
    return groups


def per_patient_loglik(result: MixtureResult, times, values, patient_ids) -> pd.DataFrame:
    """Log predictive density of each participant under their assigned cluster.

    Unassigned participants get ``NaN``.
    """

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    rows = []
    for pid, idx in _group_rows(patient_ids).items():
        k = result.assignments.get(pid, result.unassigned_label)
        if k < 0 or k >= len(result.models) or result.models[k] is None:
            rows.append({"participant_id": pid, "cluster": k, "loglik": np.nan})
            continue
        ll, _ = patient_loglik(result.models[k], result.likelihoods[k], times[idx], values[idx])
        rows.append({"participant_id": pid, "cluster": k, "loglik": ll})
    return pd.DataFrame(rows)


def count_parameters(result: MixtureResult) -> int:
    """Number of free hyperparameters across fitted clusters.

    ``model.parameters()`` already includes the likelihood's noise terms, so
    they are counted once.
    """

    return int(
        sum(p.numel() for m in result.models if m is not None for p in m.parameters() if p.requires_grad)
    )


def information_criteria(result: MixtureResult, times, values, patient_ids) -> dict[str, float]:
    """Total log-likelihood of assigned participants and an approximate BIC.

    ``bic = -2 * loglik + n_params * log(n_values)`` where ``n_values`` is the
    number of observed outcome values of assigned participants. Cluster
    labels are not counted as parameters, so this is an approximation.
    """

    ll = per_patient_loglik(result, times, values, patient_ids)
    total_ll = float(ll["loglik"].sum(skipna=True))
    labels = result.labels_for(np.asarray(patient_ids))
    n_values = int((labels != result.unassigned_label).sum() * np.asarray(values).shape[1])
    n_params = count_parameters(result)
    bic = float(-2.0 * total_ll + n_params * np.log(n_values)) if n_values > 0 else float("nan")
    return {"total_loglik": total_ll, "n_parameters": n_params, "n_values": n_values, "bic": bic}


def coverage_and_gaps(
    times,
    patient_ids,
    assignments: dict,
    t_min: float = 0.0,
    t_max: float = 25.0,
    bin_width: float = 2.0,
    min_patients_per_bin: int = 2,
    unassigned_label: int = -1,
) -> dict[str, Any]:
    """How well each cluster's members cover the time axis.

    The interval ``[t_min, t_max]`` is split into bins; a bin is covered for a
    cluster when at least ``min_patients_per_bin`` of its members have a visit
    in it. Reports the covered fraction and the longest uncovered run (years).
    """

    bins = np.arange(t_min, t_max + bin_width, bin_width)
    n_bins = len(bins) - 1
    cluster_bins: dict[int, list[set]] = defaultdict(lambda: [set() for _ in range(n_bins)])

    for t, pid in zip(np.asarray(times, dtype=float), np.asarray(patient_ids)):
        k = int(assignments.get(pid, unassigned_label))
        if k == unassigned_label or t < t_min or t > t_max:
            continue
        b = min(max(int(np.floor((t - t_min) / bin_width)), 0), n_bins - 1)
        cluster_bins[k][b].add(pid)

    per_cluster = {}
    for k, bin_sets in sorted(cluster_bins.items()):
        covered = np.array([len(s) for s in bin_sets]) >= min_patients_per_bin
        longest = run = 0
        for ok in covered:
            run = 0 if ok else run + 1
            longest = max(longest, run)
        per_cluster[k] = {
            "coverage": float(covered.mean()),
            "max_gap_years": float(longest * bin_width),
            "covered_bins": int(covered.sum()),
            "n_bins": int(n_bins),
        }

    if not per_cluster:
        return {"coverage_mean": 0.0, "coverage_min": 0.0, "max_gap_years_max": float(t_max - t_min),
                "per_cluster": {}}
    cov = [v["coverage"] for v in per_cluster.values()]
    gaps = [v["max_gap_years"] for v in per_cluster.values()]
    return {
        "coverage_mean": float(np.mean(cov)),
        "coverage_min": float(np.min(cov)),
        "max_gap_years_max": float(np.max(gaps)),
        "per_cluster": per_cluster,
    }


def roughness(result: MixtureResult, t_min: float = 0.0, t_max: float = 25.0, num_points: int = 301) -> dict:
    """Sum of squared second differences of each cluster's mean trajectory."""

    grid = np.linspace(t_min, t_max, num_points)
    by_cluster = {}
    for k in result.fitted_clusters:
        mean, _ = predict_trajectory(result.models[k], result.likelihoods[k], grid, include_noise=False)
        if mean is not None:
            by_cluster[k] = float(np.sum((mean[2:] - 2 * mean[1:-1] + mean[:-2]) ** 2))
    values = list(by_cluster.values())
    return {
        "by_cluster": by_cluster,
        "max": float(np.max(values)) if values else float("nan"),
        "mean": float(np.mean(values)) if values else float("nan"),
    }


def summary_metrics(
    result: MixtureResult,
    times,
    values,
    patient_ids,
    t_min: float = 0.0,
    t_max: float = 25.0,
    bin_width: float = 2.0,
    min_patients_per_bin: int = 2,
) -> dict[str, Any]:
    """All diagnostics in one JSON-serialisable dictionary."""

    out: dict[str, Any] = {
        "n_participants": len(result.assignments),
        "n_unassigned": int(sum(v == result.unassigned_label for v in result.assignments.values())),
        "cluster_sizes": result.cluster_sizes(),
        "n_em_iters": result.n_em_iters,
        "converged": result.converged,
    }
    out.update(information_criteria(result, times, values, patient_ids))
    out["coverage"] = coverage_and_gaps(
        times, patient_ids, result.assignments, t_min, t_max, bin_width, min_patients_per_bin,
        result.unassigned_label,
    )
    out["roughness"] = roughness(result, t_min, t_max)
    return out


def agreement_with_truth(result: MixtureResult, true_labels: dict, include_unassigned: bool = False) -> dict:
    """Adjusted Rand index and contingency table against known labels (e.g. simulated data)."""

    ids = [pid for pid in true_labels if pid in result.assignments]
    pred = np.array([result.assignments[pid] for pid in ids])
    truth = np.array([true_labels[pid] for pid in ids])
    if not include_unassigned:
        keep = pred != result.unassigned_label
        pred, truth = pred[keep], truth[keep]
    table = pd.crosstab(pd.Series(truth, name="true"), pd.Series(pred, name="fitted"))
    return {"ari": float(adjusted_rand_score(truth, pred)), "n": int(len(pred)), "contingency": table}


def stability_ari(assignment_runs: Sequence[dict]) -> float:
    """Mean pairwise ARI between assignments from different restarts."""

    if len(assignment_runs) < 2:
        return float("nan")
    ids = sorted(assignment_runs[0])
    labels = [np.array([run[pid] for pid in ids]) for run in assignment_runs]
    aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return float(np.mean(aris))


def select_k(
    times,
    values,
    patient_ids,
    anchor_z=None,
    k_values: Sequence[int] = (2, 3, 4, 5),
    n_restarts: int = 3,
    config: ModelConfig | None = None,
    outcome_names: list[str] | None = None,
    t_min: float = 0.0,
    t_max: float = 25.0,
) -> tuple[pd.DataFrame, dict[int, MixtureResult]]:
    """Fit the mixture for several K (with KMeans restarts) and tabulate diagnostics.

    For each K the restart with the highest total log-likelihood is kept.
    Columns: total log-likelihood, BIC, number of non-empty clusters,
    stability across restarts (mean pairwise ARI), time coverage and roughness.
    """

    base = config or ModelConfig()
    rows, best_results = [], {}
    for K in k_values:
        runs = []
        for seed in range(n_restarts):
            cfg = ModelConfig.from_dict(base.to_dict())
            cfg.mixture.n_clusters = int(K)
            cfg.mixture.init_random_state = seed
            cfg.mixture.n_restarts = 1
            res = fit_mixture(times, values, patient_ids, anchor_z, cfg, outcome_names)
            runs.append((information_criteria(res, times, values, patient_ids), res))
        best_seed = int(np.argmax([r[0]["total_loglik"] for r in runs]))
        crit, best = runs[best_seed]
        best_results[int(K)] = best
        cov = coverage_and_gaps(times, patient_ids, best.assignments, t_min, t_max,
                                unassigned_label=best.unassigned_label)
        rows.append(
            {
                "K": int(K),
                "total_loglik": crit["total_loglik"],
                "bic": crit["bic"],
                "n_parameters": crit["n_parameters"],
                "n_fitted_clusters": len(best.fitted_clusters),
                "n_unassigned": sum(v == best.unassigned_label for v in best.assignments.values()),
                "stability_ari": stability_ari([r[1].assignments for r in runs]),
                "coverage_min": cov["coverage_min"],
                "max_gap_years": cov["max_gap_years_max"],
                "roughness_max": roughness(best, t_min, t_max)["max"],
                "best_seed": best_seed,
            }
        )
    table = pd.DataFrame(rows)
    table["delta_loglik"] = table["total_loglik"].diff()
    return table, best_results
