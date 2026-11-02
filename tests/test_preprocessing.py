import numpy as np
import pandas as pd
import pytest

from mvmogp.pipeline import load_run_config
from mvmogp.preprocessing import ZScoreScaler, site_correct, wide_to_long
from mvmogp.simulate import SimulationSpec, simulate_cohort


def test_anchor_to_z_uses_training_mean_and_sd():
    df = pd.DataFrame({"a": [2.0, 4.0, 6.0], "b": [10.0, 20.0, 30.0]})
    scaler = ZScoreScaler().fit(df, ["a", "b"])
    np.testing.assert_allclose(scaler.anchor_to_z(0.0), [(0 - 4) / 2, (0 - 20) / 10])
    np.testing.assert_allclose(scaler.anchor_to_z([4.0, 30.0]), [0.0, 1.0])
    with pytest.raises(ValueError):
        scaler.anchor_to_z([0.0, 0.0, 0.0])


def test_scaler_round_trip_and_constant_column():
    df = pd.DataFrame({"a": [1.0, 3.0, 8.0]})
    scaler = ZScoreScaler()
    z = scaler.fit_transform(df, ["a"])["a_z"]
    np.testing.assert_allclose(scaler.to_raw("a_z", z), df["a"])
    np.testing.assert_allclose(scaler.to_raw("a_z", [1.0], is_sd=True), [df["a"].std()])
    with pytest.raises(ValueError):
        ZScoreScaler().fit(pd.DataFrame({"c": [1.0, 1.0]}), ["c"])


def test_wide_to_long_keeps_only_complete_visits():
    wide = pd.DataFrame(
        {
            "participant_id": ["P1", "P2"],
            "base_duration": [5.0, 10.0],
            "x_v1": [1.0, 2.0], "x_v2": [1.5, np.nan], "x_v3": [2.0, 3.0],
            "y_v1": [3.0, 4.0], "y_v2": [3.5, 4.5], "y_v3": [np.nan, 5.0],
        }
    )
    long = wide_to_long(wide, ["x", "y"])
    assert list(zip(long["participant_id"], long["visit"])) == [("P1", 1), ("P1", 2), ("P2", 1), ("P2", 3)]
    np.testing.assert_allclose(long["disease_duration"], [5.0, 6.0, 10.0, 12.0])


def test_site_correction_removes_planted_offsets():
    spec = SimulationSpec(n_sites=3, site_effect_scale=2.0)
    data, _ = simulate_cohort(200, spec, seed=1)
    corrected, offsets = site_correct(data, ["sara_total"], site_col="site", time_col="visit")
    gap_before = data.groupby("site")["sara_total"].mean().std()
    gap_after = corrected.groupby("site")["sara_total"].mean().std()
    assert set(offsets["site"]) == {"Site_2", "Site_3"}
    assert gap_after < gap_before


def test_run_config_overrides_parse_json_and_strings(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"gp": {"kernel": "rbf"}, "mixture": {"n_clusters": 4}}')
    data_cfg, model_cfg = load_run_config(
        cfg_path, ["gp.kernel=matern32", "gp.lengthscale_prior=[3, 0.5]", "gp.slope_prior_sd=null",
                   "data.anchor_raw=[0, 0, 0, 20, 20]"]
    )
    assert model_cfg.gp.kernel == "matern32"
    assert model_cfg.gp.lengthscale_prior == (3, 0.5)
    assert model_cfg.gp.slope_prior_sd is None
    assert model_cfg.mixture.n_clusters == 4
    assert data_cfg.anchor_raw == [0, 0, 0, 20, 20]
    with pytest.raises(KeyError):
        load_run_config(None, ["gp.not_a_field=1"])
