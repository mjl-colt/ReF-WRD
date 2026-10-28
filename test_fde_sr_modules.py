import torch

from models.ref_wrd_modules import (
    AdaptiveWaveletResidualModulator,
    FrequencyDecoupledPromptLearner,
    GaussianFrequencyDecomposition,
    LatentResidualCorrector,
    ReFWRDLoss,
    StationaryHaarWaveletTransform,
    wavelet_consistency_loss,
)


def test_stationary_haar_round_trip_and_alignment():
    swt = StationaryHaarWaveletTransform()
    for height, width in ((32, 40), (31, 35)):
        x = torch.randn(2, 3, height, width)
        bands = swt(x)
        assert all(band.shape == x.shape for band in bands.values())
        reconstructed = swt.inverse(bands)
        assert torch.allclose(x, reconstructed, atol=1e-6)


def test_gaussian_frequency_decomposition_reconstructs_input():
    decomposition = GaussianFrequencyDecomposition(
        channels=3,
        kernel_size=5,
        sigma=1.0,
    )
    image = torch.randn(2, 3, 31, 37)
    low, high = decomposition(image)
    assert low.shape == image.shape
    assert high.shape == image.shape
    assert torch.allclose(low + high, image, atol=1e-6)


def test_frequency_prompt_has_tokenwise_reliability_gate():
    learner = FrequencyDecoupledPromptLearner(
        clip_dim=16,
        hidden_dim=32,
        num_tokens=12,
        token_rank=4,
    )
    image = torch.randn(2, 3, 32, 32)
    prompt = torch.randn(2, 8, 16)
    condition, aux = learner(prompt, image)
    gate = aux["ref_wrd_reliability_gate"]
    assert condition.shape == prompt.shape
    assert gate.shape == (2, 8)
    assert torch.all((gate >= 0) & (gate <= 1))
    assert aux["ref_wrd_low_frequency"].shape == image.shape
    assert aux["ref_wrd_high_frequency"].shape == image.shape


def test_awrm_preserves_latent_resolution_and_produces_four_gates():
    modulator = AdaptiveWaveletResidualModulator(
        latent_channels=4,
        hidden_dim=16,
        prompt_dim=16,
    )
    raw_residual = torch.randn(2, 4, 17, 19)
    condition = torch.randn(2, 8, 16)
    residual, aux = modulator(raw_residual, condition)
    assert residual.shape == raw_residual.shape
    assert aux["ref_wrd_subband_gates"].shape == (2, 16)


def test_stage2_corrector_operates_in_detached_latent_space():
    model = LatentResidualCorrector(
        latent_channels=4,
        hidden_dim=16,
        num_blocks=2,
        prompt_dim=16,
        residual_scale=0.5,
    )
    base = torch.randn(2, 4, 16, 18, requires_grad=True)
    condition = torch.randn(2, 8, 16, requires_grad=True)
    corrected, residual, aux = model(base, condition, detach_inputs=True)
    assert corrected.shape == base.shape
    assert residual.shape == base.shape
    assert aux["ref_wrd_raw_latent_residual"].shape == base.shape
    corrected.mean().backward()
    assert base.grad is None
    assert condition.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_wcs_and_latent_residual_losses_are_finite():
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    base = torch.randn(2, 3, 32, 32)
    latent_residual = torch.randn(2, 4, 4, 4)
    loss = wavelet_consistency_loss(pred, target, base=base)
    assert torch.isfinite(loss)

    bundle = ReFWRDLoss()
    losses = bundle(
        pred,
        target,
        base=base,
        latent_residual=latent_residual,
    )
    assert set(losses) == {"loss_wcs", "loss_residual_magnitude"}
    assert all(torch.isfinite(value) for value in losses.values())
