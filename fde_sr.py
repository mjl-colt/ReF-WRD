import os
import sys
from typing import Any, Dict, Optional

import torch

sys.path.append(os.getcwd())

from fde_sr_core import FDE_SRBaseGenerator, FDE_SRBaseInference, FDE_SRRegularizer
from models.fde_sr_modules import FrequencyDecoupledPromptLearner, StageIIResidualCorrector


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _is_fde_sr_checkpoint(ckpt: Dict[str, Any]) -> bool:
    return isinstance(ckpt, dict) and ("fde_prompt" in ckpt or "fde_stage2" in ckpt)


def _extract_base_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(ckpt, dict) and "base" in ckpt:
        return ckpt["base"]
    return ckpt


class FDE_SRMixin:
    def _build_fde_modules(self, args: Any) -> None:
        clip_dim = _arg(args, "fde_clip_dim", 1024)
        image_channels = _arg(args, "fde_image_channels", 3)
        hidden_dim = _arg(args, "fde_hidden_dim", 128)
        prompt_hidden_dim = _arg(args, "fde_prompt_hidden_dim", 256)
        prompt_dropout = _arg(args, "fde_prompt_dropout", 0.0)
        prompt_init_scale = _arg(args, "fde_prompt_init_scale", 0.01)
        num_blocks = _arg(args, "fde_num_blocks", 6)
        residual_scale = _arg(args, "fde_residual_scale", 0.25)

        self.fde_prompt = FrequencyDecoupledPromptLearner(
            clip_dim=clip_dim,
            image_channels=image_channels,
            hidden_dim=prompt_hidden_dim,
            dropout=prompt_dropout,
            init_scale=prompt_init_scale,
        )
        self.fde_stage2 = StageIIResidualCorrector(
            image_channels=image_channels,
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
        missing_prompt, unexpected_prompt = self.fde_prompt.load_state_dict(
            ckpt.get("fde_prompt", {}),
            strict=strict,
        )
        missing_stage2, unexpected_stage2 = self.fde_stage2.load_state_dict(
            ckpt.get("fde_stage2", {}),
            strict=strict,
        )
        if missing_prompt or unexpected_prompt or missing_stage2 or unexpected_stage2:
            print(
                "[FDE_SR] Loaded FDE weights with non-strict differences: "
                f"prompt missing={len(missing_prompt)}, prompt unexpected={len(unexpected_prompt)}, "
                f"stage2 missing={len(missing_stage2)}, stage2 unexpected={len(unexpected_stage2)}"
            )

    def fde_trainable_parameters(self):
        for module in (self.fde_prompt, self.fde_stage2):
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def _prompt_context(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        return prompt_embeds.float().mean(dim=1)


class FDE_SR_gen(FDE_SRMixin, FDE_SRBaseGenerator):
    """FDE_SR generator with frequency-decoupled prompt and residual modules."""

    def __init__(self, args):
        super().__init__(args)
        self._build_fde_modules(args)
        self._move_fde_modules(torch.device("cuda"))

    def set_train(self):
        super().set_train()
        self.fde_prompt.train()
        self.fde_stage2.train()
        self.fde_prompt.requires_grad_(True)
        self.fde_stage2.requires_grad_(True)

        if _arg(self.args, "train_stage2_only", False):
            self.unet.requires_grad_(False)
            self.vae.requires_grad_(False)

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
            "format": "fde_sr_v1",
            "base": self._collect_base_state(),
            "fde_prompt": {k: v.detach().cpu() for k, v in self.fde_prompt.state_dict().items()},
            "fde_stage2": {k: v.detach().cpu() for k, v in self.fde_stage2.state_dict().items()},
            "fde_config": {
                "fde_clip_dim": _arg(self.args, "fde_clip_dim", 1024),
                "fde_hidden_dim": _arg(self.args, "fde_hidden_dim", 128),
                "fde_prompt_hidden_dim": _arg(self.args, "fde_prompt_hidden_dim", 256),
                "fde_num_blocks": _arg(self.args, "fde_num_blocks", 6),
                "fde_residual_scale": _arg(self.args, "fde_residual_scale", 0.25),
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
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        prompt_embeds, prompt_aux = self.fde_prompt(prompt_embeds, c_t)

        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=prompt_embeds.to(torch.float32),
        ).sample
        x_denoised = self.noise_scheduler.step(
            model_pred,
            self.timesteps,
            encoded_control,
            return_dict=True,
        ).prev_sample
        base_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        if _arg(self.args, "disable_stage2", False):
            output_image = base_image
            residual = torch.zeros_like(base_image)
            stage_aux = {}
        else:
            output_image, residual, stage_aux = self.fde_stage2(
                c_t,
                base_image,
                prompt_context=self._prompt_context(prompt_embeds),
            )

        aux = {
            "base_output": base_image,
            "fde_residual": residual,
        }
        aux.update(prompt_aux)
        aux.update(stage_aux)
        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds, aux


class FDE_SR_test(FDE_SRMixin, FDE_SRBaseInference):
    """FDE_SR inference module."""

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
        prompt_embeds = self.encode_prompt([prompt])
        prompt_embeds, _ = self.fde_prompt(prompt_embeds, lq)

        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self._predict_noise_tiled(lq_latent, prompt_embeds)
        x_denoised = self.noise_scheduler.step(
            model_pred,
            self.timesteps,
            lq_latent,
            return_dict=True,
        ).prev_sample
        base_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image, residual, aux = self.fde_stage2(
            lq,
            base_image,
            prompt_context=self._prompt_context(prompt_embeds),
        )
        aux["base_output"] = base_image
        aux["fde_residual"] = residual
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
]

FDE_SR_reg = FDE_SRRegularizer
