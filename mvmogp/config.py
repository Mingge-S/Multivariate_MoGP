"""Configuration objects for the anchored multivariate MoGP.

All tunable settings live in three dataclasses:

* ``GPConfig``      - the per-cluster multitask GP (mean, kernel, noise, anchor).
* ``TrainConfig``   - the optimiser and the shape penalties used for each GP.
* ``MixtureConfig`` - the hard-assignment EM that clusters participants.

The defaults reproduce the settings used in the manuscript. Every field can
be overridden in code, from a JSON file (see ``configs/``), or from the
command line of the numbered scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SUPPORTED_KERNELS = ("rbf", "matern12", "matern32", "matern52", "rq")
SUPPORTED_ANCHORS = ("hard", "soft", "none")


@dataclass
class GPConfig:
    """Settings for one cluster's multitask GP.

    Mean
        Each outcome d has a linear mean ``m_d(t) = a_d + b_d * t`` with a
        positive slope ``b_d``. With ``anchor="hard"`` the intercept ``a_d``
        is fixed to the anchor value; otherwise it is learned.

    Kernel
        ``K((t, d), (t', d')) = B[d, d'] * s * k(t, t')`` where ``k`` is the
        chosen base kernel, ``s`` an output scale, and ``B`` a low-rank plus
        diagonal task covariance (intrinsic coregionalisation model). With
        ``anchor="hard"`` the base kernel is multiplied by ``t * t'`` so the
        latent GP has zero variance at ``t = 0``.

    Noise
        Gaussian observation noise with one variance per outcome plus a shared
        variance, both bounded below by ``noise_min``.
    """

    # Anchoring ----------------------------------------------------------
    anchor: str = "hard"  # "hard": exact at t=0, "soft": penalty, "none": free intercept

    # Mean ----------------------------------------------------------------
    slope_init: float = 0.05  # initial slope (z-units per year)
    slope_prior_sd: float | None = 0.35  # log-scale SD of LogNormal slope prior; None = no prior
    intercept_prior_sd: float = 1.0  # SD of the Normal prior on free intercepts (anchor != "hard");
    # the prior is centred on 0 for anchor="none" and on the anchor for anchor="soft"

    # Kernel ---------------------------------------------------------------
    kernel: str = "rbf"  # one of SUPPORTED_KERNELS
    lengthscale_min: float = 4.0  # lower bound on the lengthscale (years)
    lengthscale_init: float | None = None  # None = GPyTorch default (just above the bound)
    lengthscale_prior: tuple[float, float] | None = (4.0, 0.5)  # Gamma(concentration, rate); mean 8 years
    outputscale_init: float | None = None
    outputscale_prior: tuple[float, float] | None = (2.0, 2.0)  # Gamma(concentration, rate)
    rq_alpha_init: float | None = None  # only used by the "rq" kernel
    task_rank: int = 1  # rank of the low-rank part of the task covariance B
    task_factor_init_scale: float = 0.1  # shrink factor applied to the random initial task factor

    # Noise ----------------------------------------------------------------
    noise_min: float = 1e-2  # lower bound on each noise variance (z-units^2)
    noise_init: float | None = None  # None = GPyTorch default (~0.70)
    learn_noise: bool = True  # False keeps the noise fixed at noise_init

    def __post_init__(self) -> None:
        if self.kernel not in SUPPORTED_KERNELS:
            raise ValueError(f"kernel must be one of {SUPPORTED_KERNELS}, got {self.kernel!r}.")
        if self.anchor not in SUPPORTED_ANCHORS:
            raise ValueError(f"anchor must be one of {SUPPORTED_ANCHORS}, got {self.anchor!r}.")
        if self.slope_init <= 0:
            raise ValueError("slope_init must be positive.")
        if self.lengthscale_prior is not None:
            self.lengthscale_prior = tuple(self.lengthscale_prior)
        if self.outputscale_prior is not None:
            self.outputscale_prior = tuple(self.outputscale_prior)
        if not self.learn_noise and self.noise_init is None:
            raise ValueError("Set noise_init when learn_noise is False.")


@dataclass
class TrainConfig:
    """Optimiser and penalty settings for fitting one cluster's GP.

    The loss is::

        -MLL + lambda_mono * mean(relu(-diff(mu))^2)
             + lambda_smooth * mean(diff2(mu)^2)
             [+ lambda_anchor * (mu(0) - anchor)^2   when anchor="soft"]

    where ``mu`` is the posterior mean evaluated on a regular grid.
    """

    n_iter: int = 100
    lr: float = 0.01
    lambda_mono: float = 15.0  # penalises decreasing posterior means
    lambda_smooth: float = 0.5  # penalises curvature of the posterior mean
    lambda_anchor: float = 10.0  # only used when GPConfig.anchor == "soft"
    grid_min: float = 0.0
    grid_max: float = 25.0
    grid_points: int = 200
    log_every: int = 50
    device: str = "auto"  # "auto", "cpu" or "cuda"
    seed: int | None = 0  # torch seed set before each GP fit; None = do not seed


@dataclass
class MixtureConfig:
    """Settings for the hard-assignment EM over clusters."""

    n_clusters: int = 4
    max_em_iters: int = 50
    min_cluster_size: int = 10  # smaller clusters are dissolved after each E-step
    init_random_state: int = 0  # KMeans initialisation seed (restart r uses init_random_state + r)
    n_restarts: int = 1  # independent EM runs from different KMeans seeds; best total log-likelihood kept
    allow_unassigned: bool = True  # poorly fitting participants may be labelled -1
    unassigned_label: int = -1
    trim_frac_per_cluster: float = 0.08  # at most this fraction trimmed per cluster
    trim_min_cluster_size: int = 20  # clusters smaller than this are never trimmed
    trim_abs_z_threshold: float = 1.5  # minimum mean |standardised residual| to be trimmed
    max_unassigned_frac: float = 0.08  # global cap on the unassigned fraction
    verbose: bool = True


@dataclass
class ModelConfig:
    """Bundle of the three configs, convenient for saving and loading."""

    gp: GPConfig = field(default_factory=GPConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"gp": asdict(self.gp), "train": asdict(self.train), "mixture": asdict(self.mixture)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            gp=_build(GPConfig, data.get("gp", {})),
            train=_build(TrainConfig, data.get("train", {})),
            mixture=_build(MixtureConfig, data.get("mixture", {})),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _build(cls, values: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)
