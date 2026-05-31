import torch

from models.fde_sr_modules import (
    FDE_SRLoss,
    FrequencyDecoupledPromptLearner,
    HaarWaveletTransform,
    StageIIResidualCorrector,
    wavelet_consistency_loss,
)


def test_haar_wavelet_round_trip_even():
    dwt = HaarWaveletTransform()
    x = torch.randn(2, 3, 32, 40)
    bands = dwt(x)
    y = dwt.inverse(bands)
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-6)


def test_haar_wavelet_round_trip_odd():
    dwt = HaarWaveletTransform()
    x = torch.randn(1, 3, 31, 35)
    bands = dwt(x)
    y = dwt.inverse(bands)
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-6)


def test_frequency_prompt_shape():
    learner = FrequencyDecoupledPromptLearner(clip_dim=16, hidden_dim=32)
    image = torch.randn(2, 3, 32, 32)
    prompt = torch.randn(2, 8, 16)
    out, aux = learner(prompt, image)
    assert out.shape == prompt.shape
    assert "fde_prompt_low_gate" in aux
    assert "fde_prompt_high_gate" in aux


def test_stage2_corrector_shape_and_range():
    model = StageIIResidualCorrector(hidden_dim=16, num_blocks=2, prompt_dim=16)
    lq = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    base = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    prompt_context = torch.randn(2, 16)
    refined, residual, aux = model(lq, base, prompt_context)
    assert refined.shape == base.shape
    assert residual.shape == base.shape
    assert refined.min() >= -1
    assert refined.max() <= 1
    assert "fde_residual" in aux


def test_fde_sr_losses_are_finite():
    pred = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    target = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    lq = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    base = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    loss = wavelet_consistency_loss(pred, target, lq=lq)
    assert torch.isfinite(loss)

    bundle = FDE_SRLoss()
    losses = bundle(pred, target, lq=lq, base=base)
    assert all(torch.isfinite(v) for v in losses.values())
