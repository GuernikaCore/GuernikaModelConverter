#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools.convert import (
    convert_t2i_adapter, convert_controlnet, convert_text_encoder, convert_vae, convert_unet, convert_safety_checker,
)
from guernikatools.models import attention
from guernikatools.utils import utils, merge_lora

import json
import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForInpainting
from diffusers import T2IAdapter, StableDiffusionAdapterPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tempfile
import requests
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)


from transformers.models.clip import modeling_clip

# Copied from https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/clip/modeling_clip.py#L677C1-L692C1
def patched_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length: int = 0):
    """ Patch to replace torch.finfo(dtype).min with -1e4
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(-1e4, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

modeling_clip._make_causal_mask = patched_make_causal_mask


def check_output_size(pipe, args):
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    if args.output_h and (args.output_h % vae_scale_factor) != 0:
        raise RuntimeError(f"Invalid output height. Must be divisible by {vae_scale_factor}")
    if args.output_w and (args.output_w % vae_scale_factor) != 0:
        raise RuntimeError(f"Invalid output width. Must be divisible by {vae_scale_factor}")
                        
    if not args.output_h:
        args.output_h = pipe.unet.config.sample_size * vae_scale_factor
    if not args.output_w:
        args.output_w = pipe.unet.config.sample_size * vae_scale_factor
            
    # if using variable size shapes, take the biggest as base
    if args.multisize:
        if args.output_h != args.output_w:
            args.output_h = max(args.output_h, args.output_w)
            args.output_w = args.output_h
        args.min_output_size = int(args.output_w*0.5)
        args.max_output_size = int(args.output_w*2)
        logger.info(f"Output size will range from {args.min_output_size}x{args.min_output_size} to {args.max_output_size}x{args.max_output_size}")
    else:
        logger.info(f"Output size will be {args.output_w}x{args.output_h}")


def main(args):
    os.makedirs(args.o, exist_ok=True)
    
    base_adapter = None
    if args.t2i_adapter_version:
        logger.info(f"Initializing T2IAdapter with {args.t2i_adapter_version}..")
        base_adapter = T2IAdapter.from_pretrained(args.t2i_adapter_version, use_auth_token=True)
    
    controlnet = None
    if args.controlnet_location:
        logger.info(f"Initializing ControlNet from {args.controlnet_location}..")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_location, local_files_only=True)
    elif args.controlnet_checkpoint_location:
        logger.info(f"Initializing ControlNet from {args.controlnet_checkpoint_location}..")
        original_config_file = args.original_config_file
        temp_dir = tempfile.gettempdir()
        if original_config_file is None:
            possible_yaml = args.model_checkpoint_location.rsplit(".", 1)[0] + '.yaml'
            if os.path.exists(possible_yaml):
                original_config_file = possible_yaml
        controlnet = download_controlnet_from_original_ckpt(
            checkpoint_path=args.model_checkpoint_location,
            original_config_file=original_config_file,
            from_safetensors=args.model_checkpoint_location.endswith("safetensors")
        )
    elif args.controlnet_version:
        logger.info(f"Initializing ControlNet with {args.controlnet_version}..")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_version, use_auth_token=True)
    
    if not args.model_version:
        if controlnet:
            if controlnet.config.cross_attention_dim == 768:
                args.model_version = "CompVis/stable-diffusion-v1-4"
            elif controlnet.config.cross_attention_dim == 2048:
                args.model_version = "stabilityai/stable-diffusion-xl-base-1.0"
            else:
                args.model_version = "stabilityai/stable-diffusion-2-1-base"
        else:
            args.model_version = "CompVis/stable-diffusion-v1-4"

    if args.model_location:
        if controlnet:
            logger.info(f"Initializing StableDiffusionControlNetPipeline from {args.model_location}..")
            try:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(args.model_location, controlnet=controlnet, local_files_only=True)
            except:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(args.model_location, controlnet=controlnet, local_files_only=True)
        else:
            logger.info(f"Initializing StableDiffusionPipeline from {args.model_location}..")
            pipe = DiffusionPipeline.from_pretrained(args.model_location, local_files_only=True)
    elif args.model_checkpoint_location:
        logger.info(f"Initializing StableDiffusionPipeline from {args.model_checkpoint_location}..")
        original_config_file = args.original_config_file
        temp_dir = tempfile.gettempdir()
        if original_config_file is None:
            possible_yaml = args.model_checkpoint_location.rsplit(".", 1)[0] + '.yaml'
            if os.path.exists(possible_yaml):
                original_config_file = possible_yaml
        if original_config_file is None:
            try:
                pipe = AutoPipelineForText2Image.from_single_file(args.model_checkpoint_location, local_files_only=True)
            except:
                pipe = AutoPipelineForInpainting.from_single_file(args.model_checkpoint_location, local_files_only=True)
        else:
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path_or_dict=args.model_checkpoint_location,
                original_config_file=original_config_file,
                from_safetensors=args.model_checkpoint_location.endswith("safetensors")
            )
        if controlnet:
            pipe.controlnet = controlnet
    elif controlnet:
        logger.info(f"Initializing StableDiffusionControlNetPipeline with {args.model_version}..")
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(args.model_version, controlnet=controlnet, use_auth_token=True)
        except:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(args.model_version, controlnet=controlnet, use_auth_token=True)
    else:
        logger.info(f"Initializing AutoPipelineForText2Image with {args.model_version}..")
        pipe = AutoPipelineForText2Image.from_pretrained(args.model_version, use_auth_token=True)
        
    if args.embeddings_location:
        logger.info(f"Loading embeddings at {args.embeddings_location}")
        embeddings_files = [join(args.embeddings_location, f) for f in listdir(args.embeddings_location) if not f.startswith('.') and isfile(join(args.embeddings_location, f))]
        for file in embeddings_files:
            logger.info(f"Loading embedding: {file}")
            pipe.load_textual_inversion(file, local_files_only=True)
        args.added_vocab = pipe.tokenizer.get_added_vocab()
        logger.info(f"Added embeddings: {args.added_vocab}")
    
    args.model_is_sdxl = hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2
    # auto apply fix if converting base SDXL
    if args.model_version == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
    
    if args.loras_to_merge:
        loras_locations = [lora_info.split(":", 1)[0] for lora_info in args.loras_to_merge]
        loras_ratios = [float(lora_info.split(":", 1)[1]) for lora_info in args.loras_to_merge]
        logger.info(f"Merging LoRAs at: {loras_locations}")
        
        if args.model_is_sdxl:
            merge_lora.merge_to_sd_model(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, loras_locations, loras_ratios)
        else:
            merge_lora.merge_to_sd_model(pipe.unet, pipe.text_encoder, None, loras_locations, loras_ratios)
    
    logger.info(f"Done.")
    check_output_size(pipe, args)
    
    # Convert models
    converted_models = []
    if base_adapter is not None:
        logger.info("Converting t2i_adapter")
        convert_t2i_adapter.main(base_adapter, args)
        converted_models.append("t2i_adapter")
        logger.info("Converted t2i_adapter")
    
    if controlnet is not None:
        logger.info("Converting controlnet")
        convert_controlnet.main(pipe, args)
        converted_models.append("controlnet")
        logger.info("Converted controlnet")
    
    if pipe and args.convert_vae_encoder:
        logger.info("Converting vae_encoder")
        convert_vae.encoder(pipe, args)
        converted_models.append("vae_encoder")
        logger.info("Converted vae_encoder")
        
    if args.convert_vae_decoder:
        logger.info("Converting vae_decoder")
        convert_vae.decoder(pipe, args)
        converted_models.append("vae_decoder")
        logger.info("Converted vae_decoder")

    if args.convert_unet:
        logger.info("Converting unet")
        convert_unet.main(pipe, args)
        converted_models.append("unet")
        converted_models.append("unet_chunk1")
        converted_models.append("unet_chunk2")
        logger.info("Converted unet")
        
    if args.convert_text_encoder:
        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            logger.info("Converting text_encoder")
            convert_text_encoder.main(pipe.tokenizer, pipe.text_encoder, args=args)
            converted_models.append("text_encoder")
            logger.info("Converted text_encoder")
        if args.model_is_sdxl:
            logger.info("Converting text_encoder_2")
            convert_text_encoder.main(pipe.tokenizer_2, pipe.text_encoder_2, model_name="text_encoder_2", args=args)
            converted_models.append("text_encoder_2")
            logger.info("Converted text_encoder_2")

    if args.convert_safety_checker:
        logger.info("Converting safety_checker")
        convert_safety_checker.main(pipe, args)
        converted_models.append("safety_checker")
        logger.info("Converted safety_checker")
        
    if args.quantize_nbits is not None:
        logger.info(f"Quantizing weights to {args.quantize_nbits}-bit precision")
        utils.quantize_weights(converted_models, args)
        logger.info(f"Quantized weights to {args.quantize_nbits}-bit precision")

    if args.bundle_resources_for_guernika:
        logger.info("Bundling resources for Guernika")
        utils.bundle_resources_for_guernika(pipe, args)
        logger.info("Bundled resources for Guernika")

    if args.clean_up_mlpackages:
        logger.info("Cleaning up MLPackages")
        utils.remove_mlpackages(args)
        logger.info("MLPackages removed")


def parser_spec():
    parser = argparse.ArgumentParser()

    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--convert-vae-encoder", action="store_true")
    parser.add_argument("--convert-vae-decoder", action="store_true")
    parser.add_argument("--convert-unet", action="store_true")
    parser.add_argument("--convert-safety-checker", action="store_true")
    parser.add_argument(
        "--model-version",
        default=None,
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=stable-diffusion"
         )
    )
    parser.add_argument(
        "--model-location",
        default=None,
        help="The local pre-trained model checkpoint and configuration to restore."
    )
    parser.add_argument(
        "--embeddings-location",
        default=None,
        help="Folder with emebeddings to load into the TextEncoder."
    )
    parser.add_argument(
        "--loras-to-merge",
        nargs='+',
        default=None,
        help="LoRAs to be merged before conversion (URL:Ratio)."
    )
    parser.add_argument(
        "--model-checkpoint-location", default=None, type=str, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original-config-file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture of the CKPT.",
    )
    parser.add_argument(
        "--t2i-adapter-support",
        action="store_true",
        help="If `--t2i-adapter-support` the output model will support T2IAdapter.",
    )
    parser.add_argument(
        "--t2i-adapter-version",
        default=None,
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=adapter"
         ))
    parser.add_argument(
        "--controlnet-support",
        action="store_true",
        help="If `--controlnet-support` the output model will support ControlNet.",
    )
    parser.add_argument(
        "--controlnet-version",
        default=None,
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=controlnet"
         ))
    parser.add_argument(
        "--controlnet-location",
        default=None,
        help=
        "The local pre-trained ControlNet checkpoint and configuration to restore."
    )
    parser.add_argument(
        "--controlnet-checkpoint-location", default=None, type=str, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--compute-unit",
        choices=tuple(cu for cu in ct.ComputeUnit._member_names_),
        default="ALL"
    )
    
    parser.add_argument(
        "--multisize",
        action="store_true",
        help="If `--multisize` the output model will support variable sizes from (output size*0.5) to (output size*2)",
    )
    parser.add_argument(
        "--output-h",
        type=int,
        default=None,
        help=
        "The desired output height of the generated image. Should be multiple of vae_scale_factor. `Defaults to pipe's sample size`",
    )
    parser.add_argument(
        "--output-w",
        type=int,
        default=None,
        help=
        "The desired output width of the generated image. Should be multiple of vae_scale_factor. `Defaults to pipe's sample size`",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=tuple(ai for ai in attention.AttentionImplementations._member_names_),
        default=attention.ATTENTION_IMPLEMENTATION_IN_EFFECT.name,
        help="The enumerated implementations trade off between ANE and GPU performance",
    )
    parser.add_argument(
        "--quantize-nbits",
        default=None,
        choices=(2, 4, 6, 8),
        type=int,
        help="If specified, quantized each model to nbits precision"
    )
    parser.add_argument(
        "-o",
        default=os.getcwd(),
        help="The resulting mlpackages will be saved into this directory")
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        ("If specified, compares the outputs of original PyTorch and final CoreML models and reports PSNR in dB. ",
         "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
         ))
    parser.add_argument(
        "--chunk-unet",
        action="store_true",
        help=
        ("If specified, generates two mlpackages out of the unet model which approximately equal weights sizes. "
         "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
         ))
    parser.add_argument(
        "--precision-full",
        action="store_true",
        help=
        ("If specified, uses full precision FP32 for all models being converted. "
         "This is required for Stable Diffusion v2.X base models."
         ))

    # Guernika Resource Bundling
    parser.add_argument(
        "--bundle-resources-for-guernika",
        action="store_true",
        help=
        ("If specified, creates a resources directory compatible with Guernika. "
         "It compiles all four models and adds them to a StableDiffusionResources directory "
         "along with a `vocab.json` and `merges.txt` for the text tokenizer"))
    parser.add_argument(
        "--resources-dir-name",
        default="Resources",
        type=str,
        help="Name for the resources directory where the Guernika model will be located.")
    parser.add_argument(
        "--clean-up-mlpackages",
        action="store_true",
        help="Removes mlpackages after a successful convesion.")
    parser.add_argument(
        "--text-encoder-vocabulary-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
        help="The URL to the vocabulary file use by the text tokenizer")
    parser.add_argument(
        "--text-encoder-merges-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
        help="The URL to the merged pairs used in by the text tokenizer.")

    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()
    
    main(args)
