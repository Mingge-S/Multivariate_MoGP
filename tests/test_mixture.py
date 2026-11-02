import numpy as np
import pandas as pd
import pytest

from mvmogp import ModelConfig, agreement_with_truth, fit_mixture, load_results, save_results
from mvmogp.metrics import count_parameters, information_criteria
from mvmogp.mixture import select_poor_fit
from mvmogp.pipeline import DataConfig, prepare_data
from mvmogp.simulate import simulate_cohort


@pytest.fixture(scope="module")
def fitted():
    # Demo-sized cohort and the manuscript's GP training length; much shorter
    # training (e.g. 40 iterations) leaves hyperparameters near their initial
    # values and recovers the planted subtypes poorly. Takes ~1 min on a GPU.
    data, truth = simulate_cohort(150, seed=0)
    prep = prepare_data(data, DataConfig())
    cfg = ModelConfig()
    cfg.train.n_iter = 100
    cfg.mixture.n_clusters = 3
    cfg.mixture.max_em_iters = 15
    cfg.mixture.verbose = False
    result = fit_mixture(prep["times"], prep["values"], prep["patient_ids"], prep["anchor_z"], cfg,
                         outcome_names=prep["z_cols"])
    return result, prep, truth


def test_recovers_planted_subtypes(fitted):
    result, _, truth = fitted
    agreement = agreement_with_truth(result, dict(zip(truth["participant_id"], truth["true_cluster"])))
    assert agreement["ari"] > 0.9


def test_trajectories_start_at_anchor(fitted):
    result, prep, _ = fitted
    traj = result.predict(0.0, 10.0, 11, band="latent")
    at_zero = traj[traj["time"] == 0.0].groupby("outcome")["mean"].first()
    expected = dict(zip(prep["z_cols"], prep["anchor_z"]))
    for outcome, value in at_zero.items():
        assert value == pytest.approx(expected[outcome], abs=1e-4)


def test_parameter_count_does_not_double_count_noise(fitted):
    result, prep, _ = fitted
    d = len(prep["z_cols"])
    # per cluster: d slopes + lengthscale + outputscale + d task factors + d task variances + d+1 noises
    per_cluster = d + 2 + d + d + (d + 1)
    assert count_parameters(result) == per_cluster * len(result.fitted_clusters)
    crit = information_criteria(result, prep["times"], prep["values"], prep["patient_ids"])
    assert np.isfinite(crit["bic"])


def test_save_and_load_round_trip(fitted, tmp_path):
    result, prep, _ = fitted
    save_results(tmp_path, result, data=prep["data"], scaler=prep["scaler"])
    loaded = load_results(tmp_path)
    before = result.predict(0, 20, 7)
    after = loaded["result"].predict(0, 20, 7)
    # The fit may run on a GPU while load_results maps to CPU, so allow float32 differences.
    np.testing.assert_allclose(before[["mean", "sd"]].to_numpy(), after[["mean", "sd"]].to_numpy(), rtol=1e-3,
                               atol=1e-3)
    assert loaded["result"].assignments == result.assignments
    assert loaded["scaler"].means == prep["scaler"].means
    assert (tmp_path / "trajectories_latent_raw.csv").exists()


def test_poor_fit_selection_respects_threshold_and_cap():
    from mvmogp.config import MixtureConfig

    diag = pd.DataFrame(
        {
            "participant_id": [f"P{i}" for i in range(40)],
            "best_cluster": [0] * 25 + [1] * 15,
            "best_ll": np.zeros(40),
            "ll_margin": np.ones(40),
            "mean_abs_z": [3.0, 2.0, 1.6] + [0.5] * 22 + [5.0] * 15,
        }
    )
    cfg = MixtureConfig(trim_frac_per_cluster=0.08, trim_min_cluster_size=20, trim_abs_z_threshold=1.5,
                        max_unassigned_frac=1.0)
    # cluster 0: floor(0.08 * 25) = 2 worst above 1.5; cluster 1 is below trim_min_cluster_size
    assert select_poor_fit(diag, cfg, 40) == {"P0", "P1"}
    cfg.max_unassigned_frac = 0.03  # cap = floor(1.2) = 1
    assert select_poor_fit(diag, cfg, 40) == {"P0"}
