import os
import sys
from typing import Any, Dict, Optional

import torch

sys.path.append(os.getcwd())

from fde_sr_core import FDE_SRBaseGenerator, FDE_SRBaseInference, FDE_SRRegularizer
from models.ref_wrd_modules import (
    FrequencyDecoupledPromptLearner,
    LatentResidualCorrector,
)


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _is_fde_sr_checkpoint(ckpt: Dict[str, Any]) -> bool:
    return isinstance(ckpt, dict) and any(
        key in ckpt
        for key in (
            "ref_wrd_prompt",
            "ref_wrd_stage2",
            "fde_prompt",
            "fde_stage2",
        )
    )


def _extract_base_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(ckpt, dict) and "base" in ckpt:
        return ckpt["base"]
    return ckpt


def _load_compatible_state(
    module: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    strict: bool,
):
    """Load new checkpoints strictly and old prototypes by matching shape."""
    if strict:
        return module.load_state_dict(state, strict=True)
    current = module.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    return module.load_state_dict(compatible, strict=False)


class FDE_SRMixin:
    def _build_fde_modules(self, args: Any) -> None:
        clip_dim = _arg(args, "fde_clip_dim", 1024)
        image_channels = _arg(args, "fde_image_channels", 3)
        hidden_dim = _arg(args, "fde_hidden_dim", 128)
        prompt_hidden_dim = _arg(args, "fde_prompt_hidden_dim", 256)
        prompt_dropout = _arg(args, "fde_prompt_dropout", 0.0)
        prompt_init_scale = _arg(args, "fde_prompt_init_scale", 0.1)
        num_tokens = _arg(args, "fde_num_tokens", 77)
        token_rank = _arg(args, "fde_token_rank", 16)
        gaussian_kernel_size = _arg(args, "fde_gaussian_kernel_size", 5)
        gaussian_sigma = _arg(args, "fde_gaussian_sigma", 1.0)
        num_blocks = _arg(args, "fde_num_blocks", 6)
        residual_scale = _arg(args, "fde_residual_scale", 0.5)
        latent_channels = _arg(args, "fde_latent_channels", 4)

        self.fde_prompt = FrequencyDecoupledPromptLearner(
            clip_dim=clip_dim,
            image_channels=image_channels,
            hidden_dim=prompt_hidden_dim,
            num_tokens=num_tokens,
            token_rank=token_rank,
            dropout=prompt_dropout,
            init_scale=prompt_init_scale,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
        )
        self.fde_stage2 = LatentResidualCorrector(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            prompt_dim=clip_dim,
            residual_scale=residual_scale,
        )

    def _move_fde_modules(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> None:
        if dtype is None:
            self.fde_prompt.to(device)
            self.fde_stage2.to(device)
        else:
            self.fde_prompt.to(device=device, dtype=dtype)
            self.fde_stage2.to(device=device, dtype=dtype)

    def _load_fde_state(self, ckpt: Optional[Dict[str, Any]], strict: bool = False) -> None:
        if not ckpt or not _is_fde_sr_checkpoint(ckpt):
            print("[FDE_SR] No FDE stage weights found; using randomly initialized FDE modules.")
            return
        prompt_state = ckpt.get("ref_wrd_prompt", ckpt.get("fde_prompt", {}))
        stage2_state = ckpt.get("ref_wrd_stage2", ckpt.get("fde_stage2", {}))
        missing_prompt, unexpected_prompt = _load_compatible_state(
            self.fde_prompt,
            prompt_state,
            strict,
        )
        missing_stage2, unexpected_stage2 = _load_compatible_state(
            self.fde_stage2,
            stage2_state,
            strict,
        )
        if missing_prompt or unexpected_prompt or missing_stage2 or unexpected_stage2:
            print(
                "[FDE_SR] Loaded FDE weights with non-strict differences: "
                f"prompt missing={len(missing_prompt)}, prompt unexpected={len(unexpected_prompt)}, "
                f"stage2 missing={len(missing_stage2)}, stage2 unexpected={len(unexpected_stage2)}"
            )

    def fde_trainable_parameters(self):
        modules = (
            (self.fde_stage2,)
            if self._training_stage() == "stage2"
            else (self.fde_prompt,)
        )
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def _training_stage(self) -> str:
        if _arg(self.args, "train_stage2_only", False):
            return "stage2"
        return _arg(self.args, "training_stage", "stage1")


class FDE_SR_gen(FDE_SRMixin, FDE_SRBaseGenerator):
    """Paper-aligned ReF-WRD generator (legacy FDE_SR entry point)."""

    def __init__(self, args):
        super().__init__(args)
        self._build_fde_modules(args)
        self._move_fde_modules(torch.device("cuda"))

    def set_train(self):
        super().set_train()
        if self._training_stage() == "stage2":
            # Stage II is deliberately trained around a fixed Stage-I latent
            # and a fixed frequency-decoupled condition.
            self.unet.eval()
            self.vae.eval()
            self.unet.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.fde_prompt.eval()
            self.fde_prompt.requires_grad_(False)
            self.fde_stage2.train()
            self.fde_stage2.requires_grad_(True)
        else:
            self.fde_prompt.train()
            self.fde_prompt.requires_grad_(True)
            self.fde_stage2.eval()
            self.fde_stage2.requires_grad_(False)

    def _collect_base_state(self) -> Dict[str, Any]:
        return {
            "vae_lora_encoder_modules": self.lora_vae_modules_encoder,
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules": self.lora_unet_others,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {
                k: v.detach().cpu()
                for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k
            },
            "state_dict_vae": {
                k: v.detach().cpu()
                for k, v in self.vae.state_dict().items()
                if "lora" in k
            },
        }

    def save_model(self, outf: str):
        sd = {
            "format": "ref_wrd_v2",
            "base": self._collect_base_state(),
            "ref_wrd_prompt": {
                k: v.detach().cpu()
                for k, v in self.fde_prompt.state_dict().items()
            },
            "ref_wrd_stage2": {
                k: v.detach().cpu()
                for k, v in self.fde_stage2.state_dict().items()
            },
            "fde_config": {
                "fde_clip_dim": _arg(self.args, "fde_clip_dim", 1024),
                "fde_hidden_dim": _arg(self.args, "fde_hidden_dim", 128),
                "fde_prompt_hidden_dim": _arg(self.args, "fde_prompt_hidden_dim", 256),
                "fde_num_tokens": _arg(self.args, "fde_num_tokens", 77),
                "fde_token_rank": _arg(self.args, "fde_token_rank", 16),
                "fde_latent_channels": _arg(self.args, "fde_latent_channels", 4),
                "fde_gaussian_kernel_size": _arg(
                    self.args, "fde_gaussian_kernel_size", 5
                ),
                "fde_gaussian_sigma": _arg(self.args, "fde_gaussian_sigma", 1.0),
                "fde_num_blocks": _arg(self.args, "fde_num_blocks", 6),
                "fde_residual_scale": _arg(self.args, "fde_residual_scale", 0.5),
            },
        }
        torch.save(sd, outf)

    def load_pretrained_base(self, ckpt_path: str, strict: bool = False) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        base = _extract_base_state(ckpt)
        missing = []
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                if n in base.get("state_dict_unet", {}):
                    p.data.copy_(base["state_dict_unet"][n].to(device=p.device, dtype=p.dtype))
                else:
                    missing.append(n)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                if n in base.get("state_dict_vae", {}):
                    p.data.copy_(base["state_dict_vae"][n].to(device=p.device, dtype=p.dtype))
                else:
                    missing.append(n)
        if strict and missing:
            raise RuntimeError(f"Missing base weights: {missing[:8]} ... total={len(missing)}")
        if missing:
            print(f"[FDE_SR] Loaded base weights with {len(missing)} missing tensors.")
        self._load_fde_state(ckpt, strict=strict)

    def forward(self, c_t, batch=None, args=None):
        stage2_training = self._training_stage() == "stage2"
        base_context = torch.no_grad() if stage2_training else torch.enable_grad()
        with base_context:
            encoded_control = (
                self.vae.encode(c_t).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
            text_prompt = self.encode_prompt(batch["prompt"])
            neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
            prompt_embeds, prompt_aux = self.fde_prompt(text_prompt, c_t)
            model_pred = self.unet(
                encoded_control,
                self.timesteps,
                encoder_hidden_states=prompt_embeds.to(torch.float32),
            ).sample
            base_latent = self.noise_scheduler.step(
                model_pred,
                self.timesteps,
                encoded_control,
                return_dict=True,
            ).prev_sample
            base_image = self.vae.decode(
                base_latent / self.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

        use_stage2 = stage2_training and not _arg(
            self.args, "disable_stage2", False
        )
        if not use_stage2:
            output_image = base_image
            corrected_latent = base_latent
            latent_residual = torch.zeros_like(base_latent)
            stage_aux = {
                "ref_wrd_latent_residual": latent_residual,
            }
        else:
            corrected_latent, latent_residual, stage_aux = self.fde_stage2(
                base_latent,
                prompt_embeds,
                detach_inputs=True,
            )
            output_image = self.vae.decode(
                corrected_latent / self.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

        aux = {
            "base_output": base_image,
            "ref_wrd_base_latent": base_latent.detach(),
            "ref_wrd_corrected_latent": corrected_latent,
            "ref_wrd_latent_residual": latent_residual,
            "ref_wrd_residual_image": output_image - base_image,
            # Legacy visualization key stays image-shaped.
            "fde_residual": output_image - base_image,
        }
        aux.update(prompt_aux)
        aux.update(stage_aux)
        aux["fde_residual"] = output_image - base_image
        return output_image, base_latent, prompt_embeds, neg_prompt_embeds, aux


class FDE_SR_test(FDE_SRMixin, FDE_SRBaseInference):
    """ReF-WRD inference module (legacy FDE_SR entry point)."""

    def __init__(self, args):
        self._pending_fde_sr_checkpoint = None
        super().__init__(args)
        self._build_fde_modules(args)
        self._move_fde_modules(self.device, dtype=self.weight_dtype)
        strict = _arg(args, "fde_strict_load", False)
        self._load_fde_state(self._pending_fde_sr_checkpoint, strict=strict)
        self.fde_prompt.eval()
        self.fde_stage2.eval()

    def load_ckpt(self, model):
        self._pending_fde_sr_checkpoint = model if _is_fde_sr_checkpoint(model) else None
        FDE_SRBaseInference.load_ckpt(self, _extract_base_state(model))

    def _predict_noise_tiled(self, lq_latent: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print("[FDE_SR Tiled Latent]: the input size is tiny and unnecessary to tile.")
            return self.unet(
                lq_latent,
                self.timesteps,
                encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
            ).sample

        print(f"[FDE_SR Tiled Latent]: input latent is {h}x{w}, running tiled UNet.")
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(dtype=lq_latent.dtype)

        grid_rows = 0
        cur_x = 0
        while cur_x < lq_latent.size(-1):
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < lq_latent.size(-2):
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_tile = lq_latent[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
                model_out = self.unet(
                    input_tile,
                    self.timesteps,
                    encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
                ).sample
                noise_preds.append(model_out)

        noise_pred = torch.zeros_like(lq_latent)
        contributors = torch.zeros_like(lq_latent)
        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                pred = noise_preds[row * grid_cols + col]
                noise_pred[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += pred * tile_weights
                contributors[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += tile_weights

        return noise_pred / contributors.clamp_min(1e-6)

    @torch.no_grad()
    def forward(self, lq, prompt, return_base: bool = False):
        text_prompt = self.encode_prompt([prompt])
        prompt_embeds, prompt_aux = self.fde_prompt(text_prompt, lq)

        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self._predict_noise_tiled(lq_latent, prompt_embeds)
        x_denoised = self.noise_scheduler.step(
            model_pred,
            self.timesteps,
            lq_latent,
            return_dict=True,
        ).prev_sample
        base_image = self.vae.decode(
            x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
        if _arg(self.args, "disable_stage2", False):
            corrected_latent = x_denoised
            latent_residual = torch.zeros_like(x_denoised)
            output_image = base_image
            aux = {}
        else:
            corrected_latent, latent_residual, aux = self.fde_stage2(
                x_denoised,
                prompt_embeds,
                detach_inputs=True,
            )
            output_image = self.vae.decode(
                corrected_latent.to(self.weight_dtype)
                / self.vae.config.scaling_factor
            ).sample.clamp(-1, 1)
        aux.update(prompt_aux)
        aux.update(
            {
                "base_output": base_image,
                "ref_wrd_base_latent": x_denoised,
                "ref_wrd_corrected_latent": corrected_latent,
                "ref_wrd_latent_residual": latent_residual,
                "ref_wrd_residual_image": output_image - base_image,
                "fde_residual": output_image - base_image,
            }
        )
        if return_base:
            return output_image, base_image, aux
        return output_image


class FDE_SR_inference_time(FDE_SR_test):
    """Inference-time benchmark variant using the same FDE_SR forward path."""

    @torch.no_grad()
    def forward(self, lq, prompt):
        return super().forward(lq, prompt, return_base=False)


__all__ = [
    "FDE_SR_gen",
    "FDE_SR_inference_time",
    "FDE_SR_reg",
    "FDE_SR_test",
    "ReF_WRD_gen",
    "ReF_WRD_inference_time",
    "ReF_WRD_reg",
    "ReF_WRD_test",
]

FDE_SR_reg = FDE_SRRegularizer
ReF_WRD_gen = FDE_SR_gen
ReF_WRD_test = FDE_SR_test
ReF_WRD_inference_time = FDE_SR_inference_time
ReF_WRD_reg = FDE_SRRegularizer
