"""Two basic figures for a fitted mixture.

* ``plot_progression_trajectories`` - one panel per outcome; each cluster's
  mean trajectory with its uncertainty band, all clusters overlaid.
* ``plot_overlay_trajectories`` - a grid of clusters (rows) by outcomes
  (columns); every participant's observed trajectory with the cluster's mean
  trajectory and uncertainty band on top.

Bands are ``mean +/- n_sd * sd``. ``band="predictive"`` includes observation
noise (where a new visit is expected); ``band="latent"`` is the uncertainty of
the mean trajectory only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mixture import MixtureResult
from .preprocessing import ZScoreScaler

# Categorical palette in fixed order: cluster i always gets slot i.
CLUSTER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
PARTICIPANT_COLOR = "#a3a19b"


def _style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def _cluster_labels(result: MixtureResult, cluster_labels: dict[int, str] | None) -> dict[int, str]:
    """Display names; by default fitted clusters are renumbered 1, 2, 3, ..."""
    if cluster_labels is not None:
        return {k: str(cluster_labels.get(k, f"Cluster {k}")) for k in result.fitted_clusters}
    return {k: f"Cluster {i + 1}" for i, k in enumerate(result.fitted_clusters)}


def _cluster_colors(result: MixtureResult, colors: Sequence[str] | dict[int, str] | None) -> dict[int, str]:
    if isinstance(colors, dict):
        return colors
    palette = list(colors or CLUSTER_COLORS)
    if len(result.fitted_clusters) > len(palette):
        raise ValueError(f"{len(result.fitted_clusters)} clusters but only {len(palette)} colours; pass colors=.")
    return {k: palette[i] for i, k in enumerate(result.fitted_clusters)}


def _trajectories(result, t_min, t_max, num_points, band, n_sd, scale, scaler) -> pd.DataFrame:
    traj = result.predict(t_min, t_max, num_points, band=band, n_sd=n_sd)
    if scale == "raw":
        if scaler is None:
            raise ValueError("scale='raw' needs the ZScoreScaler used to prepare the data.")
        traj = scaler.trajectories_to_raw(traj)
    elif scale != "z":
        raise ValueError("scale must be 'z' or 'raw'.")
    return traj


def _column_for(outcome: str, scale: str, scaler: ZScoreScaler | None) -> str:
    return scaler.raw_col_for(outcome) if scale == "raw" else outcome


def _label_for(outcome: str, column: str, outcome_labels: dict[str, str] | None) -> str:
    labels = outcome_labels or {}
    return labels.get(outcome, labels.get(column, column))


def _save(fig, save_path, dpi) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_progression_trajectories(
    result: MixtureResult,
    outcomes: Sequence[str] | None = None,
    outcome_labels: dict[str, str] | None = None,
    t_min: float = 0.0,
    t_max: float = 25.0,
    num_points: int = 200,
    band: str = "predictive",
    n_sd: float = 2.0,
    show_band: bool = True,
    band_alpha: float = 0.15,
    scale: str = "z",
    scaler: ZScoreScaler | None = None,
    cluster_labels: dict[int, str] | None = None,
    colors: Sequence[str] | dict[int, str] | None = None,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (4.0, 3.4),
    time_label: str = "Disease duration (years)",
    save_path: str | Path | None = None,
    dpi: int = 300,
):
    """Mean trajectory +/- band of every cluster, one panel per outcome.

    Parameters
    ----------
    result : fitted ``MixtureResult``.
    outcomes : subset/order of ``result.outcome_names`` to draw (default all).
    outcome_labels : display names keyed by outcome (z or raw column name).
    band, n_sd, show_band, band_alpha : uncertainty band settings.
    scale : ``"z"`` (model units) or ``"raw"`` (needs ``scaler``).
    cluster_labels : ``{cluster index: name}``; default renumbers 1..K.
    colors : list (in cluster order) or ``{cluster index: colour}``.

    Returns ``(fig, axes)`` with ``axes`` a flat array of panels.
    """

    outcomes = list(outcomes or result.outcome_names)
    traj = _trajectories(result, t_min, t_max, num_points, band, n_sd, scale, scaler)
    names = _cluster_labels(result, cluster_labels)
    palette = _cluster_colors(result, colors)

    ncols = ncols or min(len(outcomes), 5)
    nrows = int(np.ceil(len(outcomes) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), sharex=True, squeeze=False
    )
    axes = axes.ravel()

    for ax, outcome in zip(axes, outcomes):
        column = _column_for(outcome, scale, scaler)
        sub = traj[traj["outcome"] == column]
        for k in result.fitted_clusters:
            gp = sub[sub["cluster"] == k]
            if gp.empty:
                continue
            if show_band:
                ax.fill_between(gp["time"], gp["lower"], gp["upper"], color=palette[k], alpha=band_alpha,
                                linewidth=0)
            ax.plot(gp["time"], gp["mean"], color=palette[k], linewidth=2.0, label=names[k],
                    solid_capstyle="round")
        _style_axis(ax)
        ax.set_title(_label_for(outcome, column, outcome_labels), color=INK)
        ax.set_xlim(t_min, t_max)
        ax.set_xlabel(time_label, color=INK)
        ax.set_ylabel("Score" if scale == "raw" else "z-score", color=INK)
    for ax in axes[len(outcomes):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    band_text = f"band: {band} mean ± {n_sd:g} SD" if show_band else None
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if band_text:
        fig.text(0.995, 0.005, band_text, ha="right", va="bottom", fontsize=8, color=INK_MUTED)
    _save(fig, save_path, dpi)
    return fig, axes


def plot_overlay_trajectories(
    result: MixtureResult,
    data: pd.DataFrame,
    id_col: str = "participant_id",
    time_col: str = "disease_duration",
    outcomes: Sequence[str] | None = None,
    outcome_labels: dict[str, str] | None = None,
    t_min: float = 0.0,
    t_max: float = 25.0,
    num_points: int = 200,
    band: str = "predictive",
    n_sd: float = 2.0,
    show_band: bool = True,
    band_alpha: float = 0.2,
    scale: str = "z",
    scaler: ZScoreScaler | None = None,
    cluster_labels: dict[int, str] | None = None,
    colors: Sequence[str] | dict[int, str] | None = None,
    participant_color: str = PARTICIPANT_COLOR,
    participant_alpha: float = 0.6,
    share_y: bool = True,
    panel_size: tuple[float, float] = (3.6, 2.8),
    time_label: str = "Disease duration (years)",
    save_path: str | Path | None = None,
    dpi: int = 300,
):
    """Observed participant trajectories with the cluster GP overlaid.

    Rows are clusters and columns are outcomes. Each participant's visits are
    drawn as a thin connected line in ``participant_color``; the cluster's
    mean trajectory and band are drawn on top in the cluster colour.
    Participants are placed in rows by ``result.assignments`` (unassigned
    participants are not drawn).

    ``data`` needs ``id_col``, ``time_col`` and the outcome columns in the
    chosen ``scale`` (z-columns for ``"z"``, raw columns for ``"raw"``).
    Returns ``(fig, axes)`` with ``axes`` of shape ``(n_clusters, n_outcomes)``.
    """

    outcomes = list(outcomes or result.outcome_names)
    traj = _trajectories(result, t_min, t_max, num_points, band, n_sd, scale, scaler)
    names = _cluster_labels(result, cluster_labels)
    palette = _cluster_colors(result, colors)
    columns = [_column_for(o, scale, scaler) for o in outcomes]
    missing = [c for c in [id_col, time_col, *columns] if c not in data.columns]
    if missing:
        raise KeyError(f"data is missing columns: {missing}")

    df = data.copy()
    df["_cluster"] = result.labels_for(df[id_col])
    clusters = [k for k in result.fitted_clusters if (df["_cluster"] == k).any()]
    if not clusters:
        raise ValueError("No participant in data belongs to a fitted cluster.")

    fig, axes = plt.subplots(
        len(clusters), len(outcomes),
        figsize=(panel_size[0] * len(outcomes), panel_size[1] * len(clusters)),
        sharex=True, sharey="col" if share_y else False, squeeze=False,
    )

    for row, k in enumerate(clusters):
        df_k = df[df["_cluster"] == k].sort_values([id_col, time_col])
        n_members = df_k[id_col].nunique()
        for col, (outcome, column) in enumerate(zip(outcomes, columns)):
            ax = axes[row, col]
            for _, df_p in df_k.groupby(id_col, sort=False):
                x = df_p[time_col].to_numpy(dtype=float)
                y = df_p[column].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                ax.plot(x[ok], y[ok], color=participant_color, alpha=participant_alpha, linewidth=0.8,
                        marker="o", markersize=2.5, zorder=1)
            gp = traj[(traj["cluster"] == k) & (traj["outcome"] == column)]
            if not gp.empty:
                if show_band:
                    ax.fill_between(gp["time"], gp["lower"], gp["upper"], color=palette[k], alpha=band_alpha,
                                    linewidth=0, zorder=2)
                ax.plot(gp["time"], gp["mean"], color=palette[k], linewidth=2.0, zorder=3)
            _style_axis(ax)
            ax.set_xlim(t_min, t_max)
            if row == 0:
                ax.set_title(_label_for(outcome, column, outcome_labels), color=INK)
            if row == len(clusters) - 1:
                ax.set_xlabel(time_label, color=INK)
            if col == 0:
                ax.set_ylabel(f"{names[k]} (n={n_members})\n" + ("Score" if scale == "raw" else "z-score"),
                              color=INK)

    fig.tight_layout()
    if show_band:
        fig.text(0.995, 0.002, f"band: {band} mean ± {n_sd:g} SD", ha="right", va="bottom", fontsize=8,
                 color=INK_MUTED)
    _save(fig, save_path, dpi)
    return fig, axes
