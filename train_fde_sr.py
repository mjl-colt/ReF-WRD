import argparse
import os
from pathlib import Path

import diffusers
import lpips
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
from tqdm.auto import tqdm

from dataloaders.realsr_dataset import PairedSROnlineTxtDataset
from fde_sr import FDE_SR_gen, FDE_SR_reg
from models.ref_wrd_modules import ReFWRDLoss


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("List elements should be floats") from exc


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("List elements should be integers") from exc


def parse_str_list(arg):
    return arg.split(",")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--output_dir", default="experience/fde_sr")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="train_fde_sr")

    parser.add_argument("--dataset_txt_paths_list", type=parse_str_list, default=["YOUR TXT FILE PATH"])
    parser.add_argument("--dataset_prob_paths_list", type=parse_int_list, default=[1])
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--resume_fde_sr_path", default=None, type=str)

    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--lambda_fde_wavelet", default=1.0, type=float)
    parser.add_argument("--lambda_fde_residual", default=0.5, type=float)
    parser.add_argument("--lambda_fde_high", default=0.0, type=float)
    parser.add_argument("--lambda_stage2_rec", default=1.0, type=float)
    parser.add_argument("--lambda_stage2_lpips", default=0.2, type=float)
    parser.add_argument("--wavelet_ll_weight", default=0.2, type=float)
    parser.add_argument("--wavelet_high_weight", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--ram_path", type=str, default=None)

    parser.add_argument("--fde_clip_dim", default=1024, type=int)
    parser.add_argument("--fde_image_channels", default=3, type=int)
    parser.add_argument("--fde_hidden_dim", default=128, type=int)
    parser.add_argument("--fde_prompt_hidden_dim", default=256, type=int)
    parser.add_argument("--fde_prompt_dropout", default=0.0, type=float)
    parser.add_argument("--fde_prompt_init_scale", default=0.1, type=float)
    parser.add_argument("--fde_num_tokens", default=77, type=int)
    parser.add_argument("--fde_token_rank", default=16, type=int)
    parser.add_argument("--fde_gaussian_kernel_size", default=5, type=int)
    parser.add_argument("--fde_gaussian_sigma", default=1.0, type=float)
    parser.add_argument("--fde_latent_channels", default=4, type=int)
    parser.add_argument("--fde_num_blocks", default=6, type=int)
    parser.add_argument("--fde_residual_scale", default=0.5, type=float)
    parser.add_argument(
        "--training_stage",
        choices=["stage1", "stage2"],
        default="stage1",
    )
    # Retained for old scripts; it is normalized to --training_stage stage2.
    parser.add_argument("--train_stage2_only", action="store_true")
    parser.add_argument("--disable_stage2", action="store_true")

    args = parser.parse_args(input_args)
    if args.train_stage2_only:
        args.training_stage = "stage2"
    return args


def collect_generator_params(model_gen: FDE_SR_gen, args):
    layers_to_opt = []
    if args.training_stage == "stage1":
        for name, param in model_gen.unet.named_parameters():
            if param.requires_grad and ("lora" in name or "conv_in" in name):
                layers_to_opt.append(param)
        for name, param in model_gen.vae.named_parameters():
            if param.requires_grad and "lora" in name:
                layers_to_opt.append(param)
    layers_to_opt += list(model_gen.fde_trainable_parameters())
    return layers_to_opt


def collect_reg_params(model_reg: FDE_SR_reg):
    layers_to_opt_reg = []
    for name, param in model_reg.unet_update.named_parameters():
        if "lora" in name and param.requires_grad:
            layers_to_opt_reg.append(param)
    return layers_to_opt_reg


def maybe_enable_xformers(model_gen, model_reg, args):
    if not args.enable_xformers_memory_efficient_attention:
        return
    if not is_xformers_available():
        raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    model_gen.unet.enable_xformers_memory_efficient_attention()
    if model_reg is not None:
        model_reg.unet_fix.enable_xformers_memory_efficient_attention()
        model_reg.unet_update.enable_xformers_memory_efficient_attention()


def maybe_enable_gradient_checkpointing(model_gen, model_reg, args):
    if not args.gradient_checkpointing:
        return
    model_gen.unet.enable_gradient_checkpointing()
    if model_reg is not None:
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()


def make_ram_captioner(args, weight_dtype):
    from ram import inference_ram as inference
    from ram.models.ram_lora import ram

    ram_transforms = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model_vlm = ram(
        pretrained=args.ram_path,
        pretrained_condition=None,
        image_size=384,
        vit="swin_l",
    )
    model_vlm.eval()
    model_vlm.to("cuda", dtype=weight_dtype)
    return model_vlm, ram_transforms, inference


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = FDE_SR_gen(args)
    model_gen.set_train()
    if args.resume_fde_sr_path:
        model_gen.load_pretrained_base(args.resume_fde_sr_path, strict=False)

    use_vsd = args.training_stage == "stage1" and (
        args.lambda_vsd > 0 or args.lambda_vsd_lora > 0
    )
    model_reg = None
    if use_vsd:
        model_reg = FDE_SR_reg(args=args, accelerator=accelerator)
        model_reg.set_train()

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    ref_wrd_loss = ReFWRDLoss(
        lambda_wavelet=args.lambda_fde_wavelet,
        lambda_residual=args.lambda_fde_residual,
        lambda_high=args.lambda_fde_high,
        wavelet_ll_weight=args.wavelet_ll_weight,
        wavelet_high_weight=args.wavelet_high_weight,
    ).cuda()

    model_gen.vae.set_adapter(["default_encoder"])
    model_gen.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    # Some PEFT versions update requires_grad while activating adapters.
    # Re-apply the paper's stage-specific freeze policy afterwards.
    model_gen.set_train()
    if model_reg is not None:
        model_reg.unet_update.set_adapter(["default_encoder", "default_decoder", "default_others"])

    maybe_enable_xformers(model_gen, model_reg, args)
    maybe_enable_gradient_checkpointing(model_gen, model_reg, args)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    layers_to_opt = collect_generator_params(model_gen, args)
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_reg = None
    lr_scheduler_reg = None
    layers_to_opt_reg = []
    if model_reg is not None:
        layers_to_opt_reg = collect_reg_params(model_reg)
        optimizer_reg = torch.optim.AdamW(
            layers_to_opt_reg,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_reg = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    if model_reg is not None:
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
            model_gen,
            model_reg,
            optimizer,
            optimizer_reg,
            dl_train,
            lr_scheduler,
            lr_scheduler_reg,
        )
    else:
        model_gen, optimizer, dl_train, lr_scheduler = accelerator.prepare(
            model_gen,
            optimizer,
            dl_train,
            lr_scheduler,
        )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_vlm, ram_transforms, inference = make_ram_captioner(args, weight_dtype)

    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    for _epoch in range(0, args.num_training_epochs):
        for _step, batch in enumerate(dl_train):
            accumulate_models = [model_gen] if model_reg is None else [model_gen, model_reg]
            with accelerator.accumulate(*accumulate_models):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                caption = inference(x_tgt_ram.to(dtype=weight_dtype), model_vlm)
                batch["prompt"] = [f"{each_caption}" for each_caption in caption]

                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds, aux = model_gen(
                    x_src,
                    batch=batch,
                    args=args,
                )

                if args.training_stage == "stage1":
                    loss_reconstruction = F.mse_loss(
                        x_tgt_pred.float(),
                        x_tgt.float(),
                        reduction="mean",
                    ) * args.lambda_l2
                    loss_lpips = net_lpips(
                        x_tgt_pred.float(), x_tgt.float()
                    ).mean() * args.lambda_lpips
                    ref_wrd_losses = {}
                else:
                    # The Stage-I path and FDPL are frozen by set_train().
                    # Stage II follows the paper objective: image L1 + LPIPS
                    # + WCS + an L1 penalty on the latent residual itself.
                    loss_reconstruction = F.l1_loss(
                        x_tgt_pred.float(),
                        x_tgt.float(),
                        reduction="mean",
                    ) * args.lambda_stage2_rec
                    loss_lpips = net_lpips(
                        x_tgt_pred.float(), x_tgt.float()
                    ).mean() * args.lambda_stage2_lpips
                    ref_wrd_losses = ref_wrd_loss(
                        x_tgt_pred.float(),
                        x_tgt.float(),
                        base=aux["base_output"].float(),
                        latent_residual=aux["ref_wrd_latent_residual"],
                    )
                loss_ref_wrd = sum(ref_wrd_losses.values())
                loss = loss_reconstruction + loss_lpips + loss_ref_wrd

                loss_kl = x_tgt_pred.new_tensor(0.0)
                if model_reg is not None and args.lambda_vsd > 0:
                    if torch.cuda.device_count() > 1:
                        loss_kl = model_reg.module.distribution_matching_loss(
                            latents=latents_pred,
                            prompt_embeds=prompt_embeds,
                            neg_prompt_embeds=neg_prompt_embeds,
                            args=args,
                        ) * args.lambda_vsd
                    else:
                        loss_kl = model_reg.distribution_matching_loss(
                            latents=latents_pred,
                            prompt_embeds=prompt_embeds,
                            neg_prompt_embeds=neg_prompt_embeds,
                            args=args,
                        ) * args.lambda_vsd
                    loss = loss + loss_kl

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                loss_d = x_tgt_pred.new_tensor(0.0)
                if model_reg is not None and args.lambda_vsd_lora > 0:
                    if torch.cuda.device_count() > 1:
                        loss_d = model_reg.module.diff_loss(
                            latents=latents_pred,
                            prompt_embeds=prompt_embeds,
                            args=args,
                        ) * args.lambda_vsd_lora
                    else:
                        loss_d = model_reg.diff_loss(
                            latents=latents_pred,
                            prompt_embeds=prompt_embeds,
                            args=args,
                        ) * args.lambda_vsd_lora
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt_reg, args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_total": loss.detach().item(),
                        "loss_reconstruction": loss_reconstruction.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_kl": loss_kl.detach().item(),
                        "loss_d": loss_d.detach().item(),
                    }
                    for key, value in ref_wrd_losses.items():
                        logs[key] = value.detach().item()
                    if "fde_prompt_low_gate" in aux:
                        logs["fde_gate_low"] = aux["fde_prompt_low_gate"].float().mean().item()
                    if "fde_prompt_high_gate" in aux:
                        logs["fde_gate_high"] = aux["fde_prompt_high_gate"].float().mean().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(
                            args.output_dir,
                            "checkpoints",
                            f"ref_wrd_{global_step}.pkl",
                        )
                        accelerator.unwrap_model(model_gen).save_model(outf)

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    main(parse_args())
