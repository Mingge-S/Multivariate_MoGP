"""Per-cluster multitask GP: mean, kernel, likelihood, training and prediction.

For one cluster the latent trajectory of outcome ``d`` is::

    f_d(t) = a_d + b_d * t + g_d(t),       b_d > 0
    g(t)   ~ GP(0, B (x) s * k(t, t'))      (intrinsic coregionalisation model)
    y      = f_d(t) + eps,                  eps ~ N(0, sigma_d^2 + sigma^2)

With a *hard* anchor, ``a_d`` is fixed to the anchor value and the base kernel
is replaced by ``t * t' * k(t, t')``. The latent GP then has zero variance at
``t = 0``, so ``f_d(0)`` equals the anchor exactly.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors import GammaPrior, LogNormalPrior, NormalPrior
from gpytorch.utils.warnings import NumericalWarning
from linear_operator.utils.errors import NotPSDError

from .config import GPConfig, TrainConfig

_JITTERS = (1e-4, 1e-3, 1e-2)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Mean and kernel components
# ---------------------------------------------------------------------------


class PositiveSlopeLinearMean(gpytorch.means.Mean):
    """Linear mean ``a + b * t`` with ``b > 0``.

    If ``fixed_intercept`` is given, ``a`` is a constant buffer (hard anchor);
    otherwise ``a`` is a learned parameter with a Normal prior centred on
    ``intercept_init``.
    """

    def __init__(
        self,
        slope_init: float = 0.05,
        slope_prior_sd: float | None = 0.35,
        fixed_intercept: float | None = None,
        intercept_init: float = 0.0,
        intercept_prior_sd: float = 1.0,
    ):
        super().__init__()
        self.register_parameter("raw_slope", torch.nn.Parameter(torch.tensor(0.0)))
        self.register_constraint("raw_slope", Positive())
        self._set_slope(float(slope_init))
        if slope_prior_sd is not None:
            self.register_prior(
                "slope_prior",
                LogNormalPrior(math.log(slope_init), slope_prior_sd),
                lambda m: m.slope,
                lambda m, v: m._set_slope(v),
            )

        self.intercept_is_fixed = fixed_intercept is not None
        if self.intercept_is_fixed:
            self.register_buffer("intercept", torch.tensor(float(fixed_intercept)))
        else:
            self.register_parameter("intercept", torch.nn.Parameter(torch.tensor(float(intercept_init))))
            self.register_prior(
                "intercept_prior", NormalPrior(float(intercept_init), intercept_prior_sd), "intercept"
            )

    @property
    def slope(self) -> torch.Tensor:
        return self.raw_slope_constraint.transform(self.raw_slope)

    def _set_slope(self, value: torch.Tensor | float) -> None:
        value = torch.as_tensor(value, dtype=self.raw_slope.dtype, device=self.raw_slope.device)
        self.initialize(raw_slope=self.raw_slope_constraint.inverse_transform(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1) * self.slope + self.intercept


class InputScaledKernel(gpytorch.kernels.Kernel):
    """``k_anchor(t, t') = t * t' * k_base(t, t')``; zero variance at ``t = 0``."""

    has_lengthscale = False

    def __init__(self, base_kernel: gpytorch.kernels.Kernel):
        super().__init__()
        self.base_kernel = base_kernel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params: Any):
        base = self.base_kernel(x1, x2, diag=diag, **params)
        s1 = x1[..., 0]
        s2 = x2[..., 0]
        scale = s1 * s2 if diag else s1.unsqueeze(-1) * s2.unsqueeze(-2)
        return base * scale


def build_base_kernel(cfg: GPConfig) -> gpytorch.kernels.Kernel:
    """Stationary kernel over time with a lower-bounded lengthscale."""

    constraint = GreaterThan(cfg.lengthscale_min)
    if cfg.kernel == "rbf":
        kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint=constraint)
    elif cfg.kernel.startswith("matern"):
        nu = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}[cfg.kernel]
        kernel = gpytorch.kernels.MaternKernel(nu=nu, lengthscale_constraint=constraint)
    elif cfg.kernel == "rq":
        kernel = gpytorch.kernels.RQKernel(lengthscale_constraint=constraint)
        if cfg.rq_alpha_init is not None:
            kernel.alpha = cfg.rq_alpha_init
    else:  # guarded by GPConfig, kept for safety
        raise ValueError(f"Unknown kernel {cfg.kernel!r}")

    if cfg.lengthscale_prior is not None:
        kernel.register_prior("lengthscale_prior", GammaPrior(*cfg.lengthscale_prior), "lengthscale")
    if cfg.lengthscale_init is not None:
        if cfg.lengthscale_init <= cfg.lengthscale_min:
            raise ValueError("lengthscale_init must be greater than lengthscale_min.")
        kernel.lengthscale = cfg.lengthscale_init
    return kernel


# ---------------------------------------------------------------------------
# Model and likelihood
# ---------------------------------------------------------------------------


class MultitaskProgressionGP(gpytorch.models.ExactGP):
    """Multitask GP for one cluster, optionally anchored at ``t = 0``."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        num_tasks: int,
        cfg: GPConfig | None = None,
        anchor_z: np.ndarray | list[float] | None = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        cfg = cfg or GPConfig()
        self.cfg = cfg
        self.num_tasks = int(num_tasks)

        if cfg.anchor in ("hard", "soft"):
            if anchor_z is None:
                raise ValueError(f"anchor={cfg.anchor!r} requires anchor_z.")
            anchor_arr = np.asarray(anchor_z, dtype=float).reshape(-1)
            if anchor_arr.size != self.num_tasks:
                raise ValueError(f"anchor_z has {anchor_arr.size} values but num_tasks={self.num_tasks}.")
            self.register_buffer("anchor_z", torch.tensor(anchor_arr, dtype=torch.float32))
        else:
            anchor_arr = None
            self.anchor_z = None

        base_means = []
        for d in range(self.num_tasks):
            base_means.append(
                PositiveSlopeLinearMean(
                    slope_init=cfg.slope_init,
                    slope_prior_sd=cfg.slope_prior_sd,
                    fixed_intercept=float(anchor_arr[d]) if cfg.anchor == "hard" else None,
                    intercept_init=float(anchor_arr[d]) if cfg.anchor == "soft" else 0.0,
                    intercept_prior_sd=cfg.intercept_prior_sd,
                )
            )
        self.mean_module = gpytorch.means.MultitaskMean(base_means=base_means, num_tasks=self.num_tasks)

        time_kernel = build_base_kernel(cfg)
        if cfg.anchor == "hard":
            time_kernel = InputScaledKernel(time_kernel)
        if cfg.outputscale_prior is not None:
            scaled = gpytorch.kernels.ScaleKernel(time_kernel, outputscale_prior=GammaPrior(*cfg.outputscale_prior))
        else:
            scaled = gpytorch.kernels.ScaleKernel(time_kernel)
        if cfg.outputscale_init is not None:
            scaled.outputscale = cfg.outputscale_init

        self.covar_module = gpytorch.kernels.MultitaskKernel(scaled, num_tasks=self.num_tasks, rank=cfg.task_rank)
        with torch.no_grad():
            self.covar_module.task_covar_module.raw_var.data.fill_(0.0)
            self.covar_module.task_covar_module.covar_factor.data.mul_(cfg.task_factor_init_scale)

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x))

    # Convenience accessors for inspecting a fitted model -------------------

    def slopes(self) -> np.ndarray:
        return np.array([m.slope.item() for m in self.mean_module.base_means])

    def intercepts(self) -> np.ndarray:
        return np.array([float(m.intercept) for m in self.mean_module.base_means])

    def task_covariance(self) -> np.ndarray:
        return self.covar_module.task_covar_module.covar_matrix.to_dense().detach().cpu().numpy()

    def lengthscale(self) -> float:
        kernel = self.covar_module.data_covar_module.base_kernel
        if isinstance(kernel, InputScaledKernel):
            kernel = kernel.base_kernel
        return float(kernel.lengthscale.item())

    def outputscale(self) -> float:
        return float(self.covar_module.data_covar_module.outputscale.item())


def build_likelihood(num_tasks: int, cfg: GPConfig | None = None) -> gpytorch.likelihoods.MultitaskGaussianLikelihood:
    """Diagonal multitask noise: one variance per outcome plus a shared variance."""

    cfg = cfg or GPConfig()
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        rank=0,
        noise_constraint=GreaterThan(cfg.noise_min),
    )
    if cfg.noise_init is not None:
        # noise_init is the total noise variance per outcome, split equally
        # between the per-outcome and the shared terms.
        half = cfg.noise_init / 2.0
        if half <= cfg.noise_min:
            raise ValueError("noise_init must be more than twice noise_min.")
        likelihood.task_noises = torch.full((num_tasks,), half)
        likelihood.noise = torch.tensor([half])
    if not cfg.learn_noise:
        for p in likelihood.parameters():
            p.requires_grad_(False)
    return likelihood


def model_summary(model: MultitaskProgressionGP, likelihood) -> dict[str, Any]:
    """Learned hyperparameters of one cluster GP, as plain Python values."""

    return {
        "slopes": model.slopes().tolist(),
        "intercepts": model.intercepts().tolist(),
        "lengthscale": model.lengthscale(),
        "outputscale": model.outputscale(),
        "task_covariance": model.task_covariance().tolist(),
        "task_noises": likelihood.task_noises.detach().cpu().numpy().tolist(),
        "shared_noise": float(likelihood.noise.detach().cpu().item()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_cluster_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    gp_cfg: GPConfig | None = None,
    train_cfg: TrainConfig | None = None,
    anchor_z: np.ndarray | list[float] | None = None,
    verbose: bool = False,
):
    """Fit one cluster's multitask GP.

    Parameters
    ----------
    train_x : (N, 1) tensor of times (e.g. disease duration in years).
    train_y : (N, D) tensor of outcomes, usually z-scored.
    anchor_z : (D,) anchor values in the units of ``train_y``.

    Returns
    -------
    model, likelihood, log : the fitted GP (in eval mode), its likelihood and a
    list of dicts with the loss terms every ``train_cfg.log_every`` iterations.
    """

    gp_cfg = gp_cfg or GPConfig()
    train_cfg = train_cfg or TrainConfig()
    device = resolve_device(train_cfg.device)
    num_tasks = int(train_y.shape[-1])

    train_x = train_x.to(device=device, dtype=torch.float32).view(-1, 1)
    train_y = train_y.to(device=device, dtype=torch.float32)

    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)

    likelihood = build_likelihood(num_tasks, gp_cfg).to(device)
    model = MultitaskProgressionGP(train_x, train_y, likelihood, num_tasks, gp_cfg, anchor_z).to(device)
    anchor_t = model.anchor_z.view(1, -1) if model.anchor_z is not None else None

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=train_cfg.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    grid = torch.linspace(train_cfg.grid_min, train_cfg.grid_max, train_cfg.grid_points, device=device).view(-1, 1)
    zero_x = torch.zeros(1, 1, device=device)
    log: list[dict[str, float]] = []

    for i in range(train_cfg.n_iter):
        optimizer.zero_grad()

        # Exact marginal likelihood on the cluster's observations.
        model.train()
        likelihood.train()
        nll = -mll(model(train_x), train_y)

        # Shape penalties on the posterior mean over a regular time grid.
        model.eval()
        likelihood.eval()
        mean_grid = model(grid).mean
        first_diff = mean_grid[1:, :] - mean_grid[:-1, :]
        second_diff = mean_grid[2:, :] - 2 * mean_grid[1:-1, :] + mean_grid[:-2, :]
        mono_penalty = torch.relu(-first_diff).pow(2).mean()
        smooth_penalty = second_diff.pow(2).mean()
        if anchor_t is not None:
            anchor_penalty = (model(zero_x).mean - anchor_t).pow(2).mean()
        else:
            anchor_penalty = torch.zeros((), device=device)

        model.train()
        likelihood.train()

        loss = nll + train_cfg.lambda_mono * mono_penalty + train_cfg.lambda_smooth * smooth_penalty
        if gp_cfg.anchor == "soft":
            loss = loss + train_cfg.lambda_anchor * anchor_penalty
        loss.backward()

        last = i == train_cfg.n_iter - 1
        if i == 0 or last or (train_cfg.log_every > 0 and (i + 1) % train_cfg.log_every == 0):
            row = {
                "iter": i + 1,
                "nll": nll.item(),
                "mono_penalty": mono_penalty.item(),
                "smooth_penalty": smooth_penalty.item(),
                "anchor_penalty": anchor_penalty.item(),
                "total_loss": loss.item(),
            }
            log.append(row)
            if verbose:
                print(
                    f"    iter {row['iter']:4d}/{train_cfg.n_iter}  nll={row['nll']:.3f}  "
                    f"mono={row['mono_penalty']:.5f}  smooth={row['smooth_penalty']:.5f}  "
                    f"total={row['total_loss']:.3f}"
                )

        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood, log


# ---------------------------------------------------------------------------
# Scoring and prediction
# ---------------------------------------------------------------------------


def _model_device(model) -> torch.device:
    return model.train_inputs[0].device


def patient_loglik(model, likelihood, times, values) -> tuple[float, float]:
    """Score one participant's trajectory under a fitted cluster GP.

    Returns ``(loglik, mean_abs_z)``: the joint log predictive density of all
    the participant's observations, and the mean absolute standardised
    residual across visits and outcomes. Numerical failures return
    ``(-1e9, inf)`` so the cluster is treated as very unlikely.
    """

    model.eval()
    likelihood.eval()
    device = _model_device(model)
    x = torch.as_tensor(np.asarray(times, dtype=np.float32), device=device).view(-1, 1)
    y = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device).view(x.shape[0], -1)

    for jitter in _JITTERS:
        try:
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
                pred = likelihood(model(x))
                ll = pred.log_prob(y).item()
                sd = pred.variance.clamp_min(1e-6).sqrt()
                mean_abs_z = torch.abs((y - pred.mean) / sd).mean().item()
                return float(ll), float(mean_abs_z)
        except (NotPSDError, RuntimeError):
            continue

    warnings.warn(f"Could not score a participant with {x.shape[0]} visits; treating the cluster as very unlikely.")
    return -1e9, float("inf")


def predict_trajectory(model, likelihood, t_grid, include_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and SD on ``t_grid`` for every outcome.

    ``include_noise=True`` gives the predictive SD of a new noisy observation;
    ``False`` gives the SD of the latent (noise-free) trajectory.
    Returns two ``(T, D)`` arrays, or ``(None, None)`` on numerical failure.
    """

    model.eval()
    likelihood.eval()
    device = _model_device(model)
    t = torch.as_tensor(np.asarray(t_grid, dtype=np.float32), device=device).view(-1, 1)

    for jitter in _JITTERS:
        try:
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter), gpytorch.settings.fast_pred_var(), \
                    warnings.catch_warnings():
                # A hard-anchored GP has zero latent variance at t = 0, which
                # GPyTorch reports as a (harmless) negative-variance warning.
                warnings.simplefilter("ignore", NumericalWarning)
                pred = likelihood(model(t)) if include_noise else model(t)
                mean = pred.mean.cpu().numpy()
                var = pred.variance.cpu().numpy()
            return mean, np.sqrt(np.clip(var, 1e-12, None))
        except (NotPSDError, RuntimeError):
            continue
    return None, None
