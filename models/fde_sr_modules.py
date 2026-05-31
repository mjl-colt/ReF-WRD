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


def _std(x: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
    return x.float().std(dim=dim, keepdim=keepdim, unbiased=False).to(dtype=x.dtype)


def _resize_like(
    x: torch.Tensor,
    ref: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode)


class HaarWaveletTransform(nn.Module):
    """Differentiable 2D Haar DWT/IWT for image and latent tensors."""

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode

    def _pad_even(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        mode = self.pad_mode
        if min(x.shape[-2:]) <= 1 and mode == "reflect":
            mode = "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> TensorDict:
        original_size = torch.tensor(x.shape[-2:], device=x.device)
        x, pad = self._pad_even(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return {
            "ll": ll,
            "lh": lh,
            "hl": hl,
            "hh": hh,
            "pad_h": torch.tensor(pad[0], device=x.device),
            "pad_w": torch.tensor(pad[1], device=x.device),
            "original_size": original_size,
        }

    def inverse(
        self,
        bands: TensorDict,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        ll, lh, hl, hh = bands["ll"], bands["lh"], bands["hl"], bands["hh"]
        b, c, h, w = ll.shape
        out = ll.new_empty(b, c, h * 2, w * 2)
        out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5

        if crop_size is None and "original_size" in bands:
            crop_size = tuple(int(v) for v in bands["original_size"].tolist())
        if crop_size is not None:
            out = out[:, :, : crop_size[0], : crop_size[1]]
        return out


class FrequencyDescriptor(nn.Module):
    """Compact low/high frequency statistics used as conditioning signals."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
        self.dwt = HaarWaveletTransform()
        self.out_dim = channels * 18

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3))
        std = _std(x, dim=(2, 3))
        energy = x.abs().mean(dim=(2, 3))
        return torch.cat([mean, std, energy], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, TensorDict]:
        bands = self.dwt(x)
        ll_up = _resize_like(bands["ll"], x)
        high = x - ll_up
        desc = torch.cat(
            [
                self._stats(x),
                self._stats(bands["ll"]),
                self._stats(bands["lh"]),
                self._stats(bands["hl"]),
                self._stats(bands["hh"]),
                self._stats(high),
            ],
            dim=1,
        )
        maps = {
            "low": ll_up,
            "high": high,
            "ll": bands["ll"],
            "lh": bands["lh"],
            "hl": bands["hl"],
            "hh": bands["hh"],
        }
        return desc, maps


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrequencyDecoupledPromptLearner(nn.Module):
    """
    Injects image-dependent low/high-frequency deltas into CLIP prompt tokens.

    The shape stays [B, T, D], so the module can be dropped into FDE_SR without
    changing the UNet cross-attention interface.
    """

    def __init__(
        self,
        clip_dim: int = 1024,
        image_channels: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.descriptor = FrequencyDescriptor(channels=image_channels)
        self.low_proj = MLP(self.descriptor.out_dim, hidden_dim, clip_dim, dropout)
        self.high_proj = MLP(self.descriptor.out_dim, hidden_dim, clip_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(self.descriptor.out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
        self.low_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.high_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.token_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        image: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        if prompt_embeds.shape[-1] != self.clip_dim:
            raise ValueError(
                f"prompt dim {prompt_embeds.shape[-1]} does not match clip_dim {self.clip_dim}"
            )

        param = next(self.low_proj.parameters())
        image = image.to(device=param.device)
        prompt_embeds = prompt_embeds.to(device=param.device)
        desc, maps = self.descriptor(image.float())
        desc = desc.to(device=param.device, dtype=param.dtype)
        gate = self.gate(desc)
        low_delta = self.low_proj(desc) * gate[:, 0:1] * self.low_scale
        high_delta = self.high_proj(desc) * gate[:, 1:2] * self.high_scale
        delta = (low_delta + high_delta).to(dtype=prompt_embeds.dtype).unsqueeze(1)

        if token_mask is None:
            token_weight = torch.ones(
                prompt_embeds.shape[:2] + (1,),
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
        else:
            token_weight = token_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            if token_weight.ndim == 2:
                token_weight = token_weight.unsqueeze(-1)
        token_weight = torch.softmax(
            token_weight * self.token_temperature.clamp(min=0.05, max=20.0),
            dim=1,
        ) * token_weight.shape[1]
        out = prompt_embeds + delta * token_weight
        aux = {
            "fde_prompt_low_gate": gate[:, 0].detach(),
            "fde_prompt_high_gate": gate[:, 1].detach(),
            "fde_low_map": maps["low"].detach(),
            "fde_high_map": maps["high"].detach(),
        }
        return out, aux


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=_num_groups(channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(num_groups=_num_groups(channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.scale


class AdaptiveWaveletResidualModulator(nn.Module):
    """
    Modulates residual wavelet bands with global frequency descriptors and
    spatially varying gates before inverse DWT reconstruction.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_dim: int = 128,
        prompt_dim: int = 1024,
        affine_strength: float = 0.25,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.affine_strength = affine_strength
        self.dwt = HaarWaveletTransform()
        self.descriptor = FrequencyDescriptor(channels=image_channels)
        cond_dim = self.descriptor.out_dim + prompt_dim
        self.prompt_dim = prompt_dim
        self.affine = MLP(cond_dim, hidden_dim, image_channels * 4 * 2)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(image_channels * 3, hidden_dim, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, image_channels * 4, 1),
            nn.Sigmoid(),
        )

    def _condition(
        self,
        lq: torch.Tensor,
        base: torch.Tensor,
        prompt_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        param = next(self.affine.parameters())
        desc, _ = self.descriptor((base - lq).float())
        desc = desc.to(device=param.device, dtype=param.dtype)
        if prompt_context is None:
            prompt_context = desc.new_zeros(desc.shape[0], self.prompt_dim)
        else:
            prompt_context = prompt_context.to(device=param.device, dtype=param.dtype)
            if prompt_context.shape[-1] != self.prompt_dim:
                raise ValueError(
                    f"prompt_context dim {prompt_context.shape[-1]} does not match {self.prompt_dim}"
                )
        return torch.cat([desc, prompt_context], dim=1)

    def forward(
        self,
        residual: torch.Tensor,
        lq: torch.Tensor,
        base: torch.Tensor,
        prompt_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        param = next(self.affine.parameters())
        residual = residual.to(device=param.device, dtype=param.dtype)
        lq = lq.to(device=param.device, dtype=param.dtype)
        base = base.to(device=param.device, dtype=param.dtype)
        bands = self.dwt(residual)
        cond = self._condition(lq, base, prompt_context)
        affine = self.affine(cond).view(residual.shape[0], 2, 4, self.image_channels, 1, 1)
        gamma = 1.0 + torch.tanh(affine[:, 0]) * self.affine_strength
        beta = torch.tanh(affine[:, 1]) * self.affine_strength * 0.1

        spatial = self.spatial_gate(torch.cat([lq, base, base - lq], dim=1))
        spatial = F.interpolate(
            spatial,
            size=bands["ll"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        spatial = spatial.view(residual.shape[0], 4, self.image_channels, *bands["ll"].shape[-2:])
        spatial = 0.5 + spatial

        keys = ["ll", "lh", "hl", "hh"]
        modulated: TensorDict = {
            "original_size": bands["original_size"],
            "pad_h": bands["pad_h"],
            "pad_w": bands["pad_w"],
        }
        for idx, key in enumerate(keys):
            modulated[key] = (bands[key] * gamma[:, idx] + beta[:, idx]) * spatial[:, idx]

        out = self.dwt.inverse(modulated, crop_size=residual.shape[-2:])
        aux = {
            "fde_wavelet_gamma": gamma.detach().mean(dim=(3, 4)).flatten(1),
            "fde_wavelet_spatial": spatial.detach().mean(dim=2),
        }
        return out, aux


class StageIIResidualCorrector(nn.Module):
    """
    Predicts a correction residual after FDE_SR one-step restoration.

    Inputs are the low-quality image, FDE_SR base output, their difference, and
    high-frequency maps. The predicted residual is then wavelet-modulated.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_dim: int = 96,
        num_blocks: int = 6,
        prompt_dim: int = 1024,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.residual_scale = residual_scale
        self.freq = FrequencyDescriptor(channels=image_channels)
        in_ch = image_channels * 5
        self.stem = nn.Conv2d(in_ch, hidden_dim, 3, padding=1)
        blocks = []
        for idx in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dilation=1 + (idx % 3)))
        self.body = nn.Sequential(*blocks)
        self.down = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim) if prompt_dim > 0 else None
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=_num_groups(hidden_dim), num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, image_channels, 3, padding=1),
            nn.Tanh(),
        )
        self.modulator = AdaptiveWaveletResidualModulator(
            image_channels=image_channels,
            hidden_dim=max(64, hidden_dim),
            prompt_dim=prompt_dim,
        )

    def _high_map(self, x: torch.Tensor) -> torch.Tensor:
        _, maps = self.freq(x.float())
        return maps["high"].to(dtype=x.dtype)

    def forward(
        self,
        lq: torch.Tensor,
        base: torch.Tensor,
        prompt_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        param = next(self.stem.parameters())
        lq = lq.to(device=param.device, dtype=param.dtype)
        base = base.to(device=param.device, dtype=param.dtype)
        if lq.shape[-2:] != base.shape[-2:]:
            lq = _resize_like(lq, base)
        high_lq = self._high_map(lq)
        high_base = self._high_map(base)
        cond = torch.cat([lq, base, base - lq, high_lq, high_base], dim=1)

        feat = self.stem(cond)
        if self.prompt_proj is not None and prompt_context is not None:
            prompt_context = prompt_context.to(device=feat.device, dtype=feat.dtype)
            prompt_bias = self.prompt_proj(prompt_context).to(dtype=feat.dtype)
            feat = feat + prompt_bias[:, :, None, None]
        feat = self.body(feat)
        up_feat = self.up(self.down(feat))
        feat = feat + _resize_like(up_feat, feat)
        raw_residual = self.out(feat) * self.residual_scale
        residual, mod_aux = self.modulator(raw_residual, lq, base, prompt_context)
        refined = (base + residual).clamp(-1.0, 1.0)
        aux = {
            "fde_raw_residual": raw_residual,
            "fde_residual": residual,
            "fde_high_lq": high_lq.detach(),
            "fde_high_base": high_base.detach(),
        }
        aux.update(mod_aux)
        return refined, residual, aux


def wavelet_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lq: Optional[torch.Tensor] = None,
    ll_weight: float = 0.25,
    high_weight: float = 1.0,
    residual_weight: float = 0.5,
) -> torch.Tensor:
    dwt = HaarWaveletTransform()
    pred_b = dwt(pred.float())
    target_b = dwt(target.float())
    loss = F.l1_loss(pred_b["ll"], target_b["ll"]) * ll_weight
    for key in ("lh", "hl", "hh"):
        loss = loss + F.l1_loss(pred_b[key], target_b[key]) * high_weight

    if lq is not None and residual_weight > 0:
        if lq.shape[-2:] != pred.shape[-2:]:
            lq = _resize_like(lq, pred)
        pred_r = dwt((pred - lq).float())
        target_r = dwt((target - lq).float())
        for key in ("lh", "hl", "hh"):
            loss = loss + F.l1_loss(pred_r[key], target_r[key]) * residual_weight
    return loss


def high_frequency_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    freq = FrequencyDescriptor(channels=pred.shape[1]).to(device=pred.device)
    _, pred_maps = freq(pred.float())
    _, target_maps = freq(target.float())
    return F.l1_loss(pred_maps["high"], target_maps["high"]) * weight


def residual_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    detach_base: bool = True,
) -> torch.Tensor:
    if base.shape[-2:] != pred.shape[-2:]:
        base = _resize_like(base, pred)
    base_for_residual = base.detach() if detach_base else base
    return F.l1_loss(pred - base_for_residual, target - base_for_residual)


class FDE_SRLoss(nn.Module):
    """Aggregates the paper-style stage-II residual and wavelet supervision."""

    def __init__(
        self,
        lambda_wavelet: float = 0.5,
        lambda_residual: float = 0.25,
        lambda_high: float = 0.1,
        wavelet_ll_weight: float = 0.25,
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
        lq: Optional[torch.Tensor] = None,
        base: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        losses: TensorDict = {}
        losses["loss_fde_wavelet"] = wavelet_consistency_loss(
            pred,
            target,
            lq=lq,
            ll_weight=self.wavelet_ll_weight,
            high_weight=self.wavelet_high_weight,
        ) * self.lambda_wavelet
        if base is not None:
            losses["loss_fde_residual"] = residual_alignment_loss(pred, target, base) * self.lambda_residual
        else:
            losses["loss_fde_residual"] = pred.new_tensor(0.0)
        if self.lambda_high > 0:
            losses["loss_fde_high"] = high_frequency_l1_loss(pred, target) * self.lambda_high
        else:
            losses["loss_fde_high"] = pred.new_tensor(0.0)
        return losses
