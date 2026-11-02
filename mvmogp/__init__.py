"""Anchored multivariate mixture of Gaussian processes (MoGP) for disease progression."""

from .config import GPConfig, MixtureConfig, ModelConfig, TrainConfig
from .io import load_results, save_results
from .metrics import agreement_with_truth, information_criteria, select_k, summary_metrics
from .mixture import MixtureResult, fit_mixture
from .model import MultitaskProgressionGP, fit_cluster_gp, patient_loglik, predict_trajectory
from .plotting import plot_overlay_trajectories, plot_progression_trajectories
from .preprocessing import ZScoreScaler, site_correct, wide_to_long
from .simulate import SimulationSpec, simulate_cohort, true_trajectories

__version__ = "1.0.0"

__all__ = [
    "GPConfig",
    "MixtureConfig",
    "ModelConfig",
    "TrainConfig",
    "MixtureResult",
    "MultitaskProgressionGP",
    "SimulationSpec",
    "ZScoreScaler",
    "agreement_with_truth",
    "fit_cluster_gp",
    "fit_mixture",
    "information_criteria",
    "load_results",
    "patient_loglik",
    "plot_overlay_trajectories",
    "plot_progression_trajectories",
    "predict_trajectory",
    "save_results",
    "select_k",
    "simulate_cohort",
    "site_correct",
    "summary_metrics",
    "true_trajectories",
    "wide_to_long",
]
