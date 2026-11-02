import numpy as np
import pytest
import torch

from mvmogp.config import GPConfig, TrainConfig
from mvmogp.model import InputScaledKernel, build_likelihood, fit_cluster_gp, predict_trajectory


def _toy_data(n=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.5, 20, n))
    y = np.stack([-2.0 + 0.2 * (k + 1) * t + rng.normal(0, 0.2, n) for k in range(d)], axis=1)
    return torch.tensor(t, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


FAST = TrainConfig(n_iter=5, device="cpu")


def test_input_scaled_kernel_vanishes_at_zero():
    import gpytorch

    kernel = InputScaledKernel(gpytorch.kernels.RBFKernel())
    x = torch.tensor([[0.0], [1.0], [3.0]])
    K = kernel(x, x).to_dense()
    assert torch.allclose(K[0], torch.zeros(3))
    assert torch.allclose(kernel(x, x, diag=True)[0], torch.tensor(0.0))
    assert K[2, 2] > K[1, 1]  # variance grows with t^2


def test_hard_anchor_is_exact_at_time_zero():
    t, y = _toy_data()
    anchor = np.array([-2.0, -1.5, -3.0])
    model, lik, _ = fit_cluster_gp(t, y, GPConfig(anchor="hard"), FAST, anchor)
    mean, sd = predict_trajectory(model, lik, [0.0, 10.0], include_noise=False)
    np.testing.assert_allclose(mean[0], anchor, atol=1e-4)
    assert np.all(sd[0] < 1e-2)
    assert np.all(sd[1] > sd[0])


@pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52", "rq"])
@pytest.mark.parametrize("anchor", ["hard", "soft", "none"])
def test_every_kernel_and_anchor_mode_fits(kernel, anchor):
    t, y = _toy_data()
    model, lik, log = fit_cluster_gp(t, y, GPConfig(kernel=kernel, anchor=anchor), FAST, np.zeros(3))
    mean, sd = predict_trajectory(model, lik, np.linspace(0, 20, 5))
    assert mean.shape == (5, 3) and np.isfinite(mean).all() and np.isfinite(sd).all()
    assert np.isfinite(log[-1]["total_loss"])


def test_lengthscale_and_noise_settings_are_respected():
    cfg = GPConfig(lengthscale_min=2.0, lengthscale_init=6.0, noise_min=0.05, noise_init=0.4, learn_noise=False)
    t, y = _toy_data()
    model, lik, _ = fit_cluster_gp(t, y, cfg, FAST, np.zeros(3))
    assert model.lengthscale() >= 2.0
    np.testing.assert_allclose(lik.task_noises.detach().numpy(), 0.2, rtol=1e-5)
    np.testing.assert_allclose(lik.noise.item(), 0.2, rtol=1e-5)


def test_config_validation():
    with pytest.raises(ValueError):
        GPConfig(kernel="periodic")
    with pytest.raises(ValueError):
        GPConfig(learn_noise=False)
    with pytest.raises(ValueError):
        build_likelihood(3, GPConfig(noise_min=0.1, noise_init=0.15))
