"""Paper-aligned ReF-WRD modules.

This module implements the proposed components described in the manuscript:

* Frequency-Decoupled Prompt Learning (FDPL) with Gaussian low/high-frequency
  separation and a token-wise reliability gate.
* A latent Stage-II residual corrector around a detached Stage-I base latent.
* Adaptive Wavelet Residual Modulation (A-WRM) using a non-decimated Haar
  stationary wavelet transform.
* Wavelet Consistency Supervision (WCS), matching high-frequency subbands to
  the reference and anchoring the low-frequency subband to the Stage-I base.

Legacy FDE aliases are kept at the bottom so existing entry points and
checkpoints can migrate without an all-at-once rename.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_tokens(tokens: torch.Tensor, length: int) -> torch.Tensor:
    if tokens.shape[1] == length:
        return tokens
    return F.interpolate(
        tokens.transpose(1, 2),
        size=length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class StationaryHaarWaveletTransform(nn.Module):
    """Single-level, non-decimated Haar SWT/ISWT.

    All four subbands retain the input spatial resolution. Periodic shifts are
    used at the image boundary, and the inverse recovers the aligned reference
    phase exactly.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> TensorDict:
        x_right = torch.roll(x, shifts=-1, dims=-1)
        x_down = torch.roll(x, shifts=-1, dims=-2)
        x_diag = torch.roll(x, shifts=(-1, -1), dims=(-2, -1))
        return {
            "ll": (x + x_right + x_down + x_diag) * 0.5,
            "lh": (x - x_right + x_down - x_diag) * 0.5,
            "hl": (x + x_right - x_down - x_diag) * 0.5,
            "hh": (x - x_right - x_down + x_diag) * 0.5,
        }

    @staticmethod
    def inverse(bands: TensorDict) -> torch.Tensor:
        return (
            bands["ll"] + bands["lh"] + bands["hl"] + bands["hh"]
        ) * 0.5


class GaussianFrequencyDecomposition(nn.Module):
    """Gaussian low-pass decomposition used by FDPL."""

    def __init__(
        self,
        channels: int = 3,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coords.square()) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.channels = channels
        self.padding = kernel_size // 2
        self.register_buffer(
            "kernel",
            kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(
                channels, 1, 1, 1
            ),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} image channels, got {image.shape[1]}"
            )
        pad_mode = "reflect" if min(image.shape[-2:]) > self.padding else "replicate"
        padded = F.pad(
            image,
            (self.padding, self.padding, self.padding, self.padding),
            mode=pad_mode,
        )
        kernel = self.kernel.to(device=image.device, dtype=image.dtype)
        low = F.conv2d(padded, kernel, groups=self.channels)
        return low, image - low


class ConvFeatureBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            3,
            stride=stride,
            padding=1,
        )
        self.norm = nn.GroupNorm(_num_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class FrequencyBranchEncoder(nn.Module):
    """Independent shallow CNN extractor for one frequency branch."""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        width = max(16, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.SiLU(),
            ConvFeatureBlock(width),
            ConvFeatureBlock(width, stride=2),
            ConvFeatureBlock(width),
            ConvFeatureBlock(width, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(width, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class LowRankTokenAdapter(nn.Module):
    """Two-layer, low-rank soft-prompt adapter producing N x D tokens."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_tokens: int,
        token_dim: int,
        rank: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        rank = max(1, min(rank, token_dim))
        self.num_tokens = num_tokens
        self.rank = rank
        self.coefficients = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * rank),
        )
        self.basis = nn.Parameter(torch.empty(rank, token_dim))
        self.bias = nn.Parameter(torch.zeros(num_tokens, token_dim))
        nn.init.normal_(self.basis, std=0.02)
        nn.init.normal_(self.bias, std=0.002)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        coeff = self.coefficients(features).view(
            features.shape[0],
            self.num_tokens,
            self.rank,
        )
        return torch.einsum("bnr,rd->bnd", coeff, self.basis) + self.bias


class FrequencyDecoupledPromptLearner(nn.Module):
    """FDPL with independent frequency extractors and token-wise gating."""

    def __init__(
        self,
        clip_dim: int = 1024,
        image_channels: int = 3,
        hidden_dim: int = 256,
        num_tokens: int = 77,
        token_rank: int = 16,
        dropout: float = 0.0,
        init_scale: float = 0.1,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.num_tokens = num_tokens
        self.decompose = GaussianFrequencyDecomposition(
            channels=image_channels,
            kernel_size=gaussian_kernel_size,
            sigma=gaussian_sigma,
        )
        self.low_encoder = FrequencyBranchEncoder(image_channels, hidden_dim)
        self.high_encoder = FrequencyBranchEncoder(image_channels, hidden_dim)
        self.gate_encoder = FrequencyBranchEncoder(image_channels, hidden_dim)
        self.low_adapter = LowRankTokenAdapter(
            hidden_dim,
            hidden_dim,
            num_tokens,
            clip_dim,
            rank=token_rank,
            dropout=dropout,
        )
        self.high_adapter = LowRankTokenAdapter(
            hidden_dim,
            hidden_dim,
            num_tokens,
            clip_dim,
            rank=token_rank,
            dropout=dropout,
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tokens),
        )
        self.high_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.norm = nn.LayerNorm(clip_dim)

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        image: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        if prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [B, N, D]")
        if prompt_embeds.shape[-1] != self.clip_dim:
            raise ValueError(
                f"prompt dim {prompt_embeds.shape[-1]} does not match {self.clip_dim}"
            )

        param = next(self.low_encoder.parameters())
        image = image.to(device=param.device, dtype=param.dtype)
        prompt_embeds = prompt_embeds.to(device=param.device)
        low, high = self.decompose(image)

        low_tokens = self.low_adapter(self.low_encoder(low))
        high_tokens = self.high_adapter(self.high_encoder(high))
        gate = torch.sigmoid(self.gate_head(self.gate_encoder(image))).unsqueeze(-1)

        token_length = prompt_embeds.shape[1]
        low_tokens = _resize_tokens(low_tokens, token_length)
        high_tokens = _resize_tokens(high_tokens, token_length)
        gate = _resize_tokens(gate, token_length)

        condition = self.norm(
            low_tokens + self.high_scale * gate * high_tokens
        ).to(dtype=prompt_embeds.dtype)
        if token_mask is not None:
            mask = token_mask.to(device=condition.device, dtype=condition.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            condition = condition * mask

        aux = {
            "ref_wrd_reliability_gate": gate.squeeze(-1).detach(),
            "ref_wrd_low_frequency": low.detach(),
            "ref_wrd_high_frequency": high.detach(),
            "fde_prompt_low_gate": torch.ones_like(gate[:, :, 0]).detach(),
            "fde_prompt_high_gate": gate[:, :, 0].detach(),
            "fde_low_map": low.detach(),
            "fde_high_map": high.detach(),
        }
        return condition, aux


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.scale


class DepthwiseLowFrequencyProcessor(nn.Module):
    """Residual depth-wise 3x3 processor for the LL subband."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(F.silu(self.depthwise(x)))
        return x + update * self.scale


class ChannelAttentionResidualBlock(nn.Module):
    """Residual channel-attention block for concatenated HF subbands."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.conv2(F.silu(self.conv1(x)))
        return x + update * self.attention(update) * self.scale


class AdaptiveWaveletResidualModulator(nn.Module):
    """A-WRM operating on non-decimated latent Haar subbands."""

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_dim: int = 128,
        prompt_dim: int = 1024,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.prompt_dim = prompt_dim
        self.swt = StationaryHaarWaveletTransform()
        self.low_processor = DepthwiseLowFrequencyProcessor(latent_channels)
        high_channels = latent_channels * 3
        self.high_processor = nn.Sequential(
            ChannelAttentionResidualBlock(high_channels),
            ChannelAttentionResidualBlock(high_channels),
        )
        self.gate = nn.Sequential(
            nn.Linear(latent_channels + prompt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels * 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raw_residual: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, TensorDict]:
        if raw_residual.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, "
                f"got {raw_residual.shape[1]}"
            )
        condition_pool = condition.detach().mean(dim=1)
        residual_pool = raw_residual.mean(dim=(2, 3))
        gates = self.gate(
            torch.cat([residual_pool, condition_pool], dim=1)
        ).view(
            raw_residual.shape[0],
            4,
            self.latent_channels,
            1,
            1,
        )

        bands = self.swt(raw_residual)
        low = self.low_processor(bands["ll"])
        high = self.high_processor(
            torch.cat([bands["lh"], bands["hl"], bands["hh"]], dim=1)
        )
        high_lh, high_hl, high_hh = high.chunk(3, dim=1)
        modulated = {
            "ll": gates[:, 0] * low,
            "lh": gates[:, 1] * high_lh,
            "hl": gates[:, 2] * high_hl,
            "hh": gates[:, 3] * high_hh,
        }
        output = self.swt.inverse(modulated)
        flat_gates = gates.detach().view(gates.shape[0], -1)
        return output, {
            "ref_wrd_subband_gates": flat_gates,
            "fde_wavelet_gamma": flat_gates,
        }


class LatentResidualCorrector(nn.Module):
    """Single-pass latent residual branch followed by A-WRM."""

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        prompt_dim: int = 1024,
        residual_scale: float = 0.5,
        image_channels: Optional[int] = None,
    ):
        super().__init__()
        if image_channels is not None and latent_channels == 4:
            latent_channels = image_channels
        self.latent_channels = latent_channels
        self.residual_scale = residual_scale
        self.stem = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        self.body = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, dilation=1 + index % 3)
                for index in range(num_blocks)
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(_num_groups(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 3, padding=1),
        )
        self.modulator = AdaptiveWaveletResidualModulator(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
            prompt_dim=prompt_dim,
        )

    def forward(
        self,
        base_latent: torch.Tensor,
        condition: torch.Tensor,
        residual_scale: Optional[float] = None,
        detach_inputs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        if detach_inputs:
            base_latent = base_latent.detach()
            condition = condition.detach()
        param = next(self.stem.parameters())
        base_latent = base_latent.to(device=param.device, dtype=param.dtype)
        condition = condition.to(device=param.device, dtype=param.dtype)

        features = self.stem(base_latent)
        prompt_bias = self.prompt_proj(condition.mean(dim=1))
        features = features + prompt_bias[:, :, None, None]
        raw_residual = self.out(self.body(features))
        modulated_residual, awrm_aux = self.modulator(raw_residual, condition)
        eta = self.residual_scale if residual_scale is None else residual_scale
        corrected_latent = base_latent + float(eta) * modulated_residual
        aux = {
            "ref_wrd_raw_latent_residual": raw_residual,
            "ref_wrd_latent_residual": modulated_residual,
            "ref_wrd_residual_scale": modulated_residual.new_tensor(float(eta)),
            "fde_raw_residual": raw_residual,
            "fde_residual": modulated_residual,
        }
        aux.update(awrm_aux)
        return corrected_latent, modulated_residual, aux


def wavelet_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base: Optional[torch.Tensor] = None,
    lq: Optional[torch.Tensor] = None,
    ll_weight: float = 0.2,
    high_weight: float = 1.0,
    residual_weight: float = 0.0,
) -> torch.Tensor:
    """Paper WCS: high-frequency to target, LL to detached Stage-I base."""

    del residual_weight
    if base is None:
        base = lq
    if base is None:
        raise ValueError("WCS requires the Stage-I base output")
    swt = StationaryHaarWaveletTransform()
    pred_bands = swt(pred.float())
    target_bands = swt(target.float())
    base_bands = swt(base.detach().float())
    loss = F.l1_loss(pred_bands["ll"], base_bands["ll"]) * ll_weight
    for key in ("lh", "hl", "hh"):
        loss = loss + F.l1_loss(
            pred_bands[key],
            target_bands[key],
        ) * high_weight
    return loss


def high_frequency_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    swt = StationaryHaarWaveletTransform()
    pred_bands = swt(pred.float())
    target_bands = swt(target.float())
    return sum(
        F.l1_loss(pred_bands[key], target_bands[key])
        for key in ("lh", "hl", "hh")
    ) * weight


class ReFWRDLoss(nn.Module):
    """Stage-II WCS and latent residual-magnitude objectives."""

    def __init__(
        self,
        lambda_wavelet: float = 1.0,
        lambda_residual: float = 0.5,
        lambda_high: float = 0.0,
        wavelet_ll_weight: float = 0.2,
        wavelet_high_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_wavelet = lambda_wavelet
        self.lambda_residual = lambda_residual
        self.lambda_high = lambda_high
        self.wavelet_ll_weight = wavelet_ll_weight
        self.wavelet_high_weight = wavelet_high_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base: torch.Tensor,
        latent_residual: Optional[torch.Tensor] = None,
        lq: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        del lq
        losses: TensorDict = {
            "loss_wcs": wavelet_consistency_loss(
                pred,
                target,
                base=base,
                ll_weight=self.wavelet_ll_weight,
                high_weight=self.wavelet_high_weight,
            )
            * self.lambda_wavelet
        }
        if latent_residual is None:
            residual_penalty = (pred - base.detach()).abs().mean()
        else:
            residual_penalty = latent_residual.float().abs().mean()
        losses["loss_residual_magnitude"] = (
            residual_penalty * self.lambda_residual
        )
        if self.lambda_high > 0:
            losses["loss_high_frequency"] = (
                high_frequency_l1_loss(pred, target) * self.lambda_high
            )
        return losses


StageIIResidualCorrector = LatentResidualCorrector
FDE_SRLoss = ReFWRDLoss


__all__ = [
    "AdaptiveWaveletResidualModulator",
    "FDE_SRLoss",
    "FrequencyDecoupledPromptLearner",
    "GaussianFrequencyDecomposition",
    "LatentResidualCorrector",
    "ReFWRDLoss",
    "StageIIResidualCorrector",
    "StationaryHaarWaveletTransform",
    "high_frequency_l1_loss",
    "wavelet_consistency_loss",
]
