import argparse
import glob
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())

from fde_sr import FDE_SR_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from ram import inference_ram as inference
from ram.models.ram_lora import ram


tensor_transforms = transforms.Compose([transforms.ToTensor()])
ram_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def list_images(path):
    if os.path.isdir(path):
        image_names = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            image_names.extend(glob.glob(os.path.join(path, ext)))
        return sorted(image_names)
    return [path]


def get_validation_prompt(args, image, model, weight_dtype, device="cuda"):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    return validation_prompt, lq


def preprocess_input(input_image, args):
    ori_width, ori_height = input_image.size
    rscale = args.upscale
    resize_flag = False

    if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
        scale = (args.process_size // rscale) / min(ori_width, ori_height)
        input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)), Image.LANCZOS)
        resize_flag = True

    input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale), Image.LANCZOS)
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    return input_image, (ori_width, ori_height), resize_flag


def residual_to_pil(residual):
    residual = residual[0].detach().float().cpu()
    scale = residual.abs().amax().clamp_min(1e-6)
    residual = (residual / scale * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(residual)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", "-i", type=str, default="preset/datasets/test_dataset/input")
    parser.add_argument("--output_dir", "-o", type=str, default="preset/datasets/test_dataset/output_fde_sr")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")
    parser.add_argument("--fde_sr_path", type=str, default="preset/models/ref_wrd.pkl")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--ram_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--save_prompts", action="store_true", default=True)
    parser.add_argument("--no_save_prompts", action="store_false", dest="save_prompts")
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--merge_and_unload_lora", default=False)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

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
    parser.add_argument("--disable_stage2", action="store_true")
    parser.add_argument("--fde_strict_load", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = FDE_SR_test(args)
    image_names = list_images(args.input_image)

    DAPE = ram(
        pretrained=args.ram_path,
        pretrained_condition=args.ram_ft_path,
        image_size=384,
        vit="swin_l",
    )
    DAPE.eval()
    DAPE.to("cuda")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    DAPE = DAPE.to(dtype=weight_dtype)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, "txt")
        os.makedirs(txt_path, exist_ok=True)
    if args.save_intermediate:
        base_path = os.path.join(args.output_dir, "base")
        residual_path = os.path.join(args.output_dir, "residual")
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(residual_path, exist_ok=True)

    print(f"There are {len(image_names)} images.")
    for image_name in image_names:
        input_image = Image.open(image_name).convert("RGB")
        input_image, original_size, resize_flag = preprocess_input(input_image, args)
        bname = os.path.basename(image_name)

        validation_prompt, lq = get_validation_prompt(args, input_image, DAPE, weight_dtype)
        if args.save_prompts:
            txt_save_path = os.path.join(txt_path, f"{os.path.splitext(bname)[0]}.txt")
            with open(txt_save_path, "w", encoding="utf-8") as f:
                f.write(validation_prompt)

        print(f"process {image_name}, tag: {validation_prompt}".encode("utf-8"))

        with torch.no_grad():
            lq = lq * 2 - 1
            if args.save_intermediate:
                output_image, base_image, aux = model(lq, prompt=validation_prompt, return_base=True)
            else:
                output_image = model(lq, prompt=validation_prompt)
                base_image = None
                aux = None

            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == "adain":
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == "wavelet":
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)

            if resize_flag:
                output_pil = output_pil.resize(
                    (int(args.upscale * original_size[0]), int(args.upscale * original_size[1])),
                    Image.LANCZOS,
                )

            if args.save_intermediate and base_image is not None and aux is not None:
                base_pil = transforms.ToPILImage()(base_image[0].cpu() * 0.5 + 0.5)
                residual_pil = residual_to_pil(aux["ref_wrd_residual_image"])
                base_pil.save(os.path.join(base_path, bname))
                residual_pil.save(os.path.join(residual_path, bname))

        output_pil.save(os.path.join(args.output_dir, bname))
