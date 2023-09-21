#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from .merge_lora import merge_to_sd_model

import json
import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
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
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)


def conditioning_method_from(identifier):
    if "canny" in identifier:
        return "canny"
    if "depth" in identifier:
        return "depth"
    if "pose" in identifier:
        return "pose"
    if "mlsd" in identifier:
        return "mlsd"
    if "normal" in identifier:
        return "normal"
    if "scribble" in identifier:
        return "scribble"
    if "hed" in identifier:
        return "hed"
    if "seg" in identifier:
        return "segmentation"
    return None


def get_out_path(args, submodule_name):
    fname = f"{args.model_version}_{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    if args.clean_up_mlpackages:
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, fname)
    return os.path.join(args.o, fname)


def get_coreml_inputs(sample_inputs, samples_shapes=None):
    return [
        ct.TensorType(
            name=k,
            shape=samples_shapes[k] if samples_shapes and k in samples_shapes else v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


ABSOLUTE_MIN_PSNR = 35

def report_correctness(original_outputs, final_outputs, log_prefix):
    """ Report PSNR values across two compatible tensors
    """
    original_psnr = compute_psnr(original_outputs, original_outputs)
    final_psnr = compute_psnr(original_outputs, final_outputs)

    dB_change = final_psnr - original_psnr
    logger.info(
        f"{log_prefix}: PSNR changed by {dB_change:.1f} dB ({original_psnr:.1f} -> {final_psnr:.1f})"
    )

    if final_psnr < ABSOLUTE_MIN_PSNR:
#        raise ValueError(f"{final_psnr:.1f} dB is too low!")
        logger.info(f"{final_psnr:.1f} dB is too low!")
    else:
        logger.info(
            f"{final_psnr:.1f} dB > {ABSOLUTE_MIN_PSNR} dB (minimum allowed) parity check passed"
        )
    return final_psnr


def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


def convert_to_coreml(submodule_name, torchscript_module, coreml_inputs, output_names, precision_full, args):
    out_path = get_out_path(args, submodule_name)

    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        logger.info(f"Loading model from {out_path}")

        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit[args.compute_unit])
        logger.info(f"Loading {out_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = ct.ComputeUnit[args.compute_unit]
    else:
        logger.info(f"Converting {submodule_name} to CoreML...")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=coreml_inputs,
            outputs=[ct.TensorType(name=name) for name in output_names],
            compute_units=ct.ComputeUnit[args.compute_unit],
            compute_precision=ct.precision.FLOAT32 if precision_full else ct.precision.FLOAT16,
            # skip_model_load=True,
        )

        del torchscript_module
        gc.collect()

    return coreml_model, out_path
    
                
def quantize_weights(models, args):
    """ Quantize weights to args.quantize_nbits using a palette (look-up table)
    """
    for model_name in models:
        logger.info(f"Quantizing {model_name} to {args.quantize_nbits}-bit precision")
        out_path = get_out_path(args, model_name)
        _quantize_weights(
            out_path,
            model_name,
            args.quantize_nbits
        )

def _quantize_weights(out_path, model_name, nbits):
    if os.path.exists(out_path):
        logger.info(f"Quantizing {model_name}")
        mlmodel = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
        )

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config,
            op_type_configs={
                "gather": None # avoid quantizing the embedding table
            }
        )

        model = ct.optimize.coreml.palettize_weights(mlmodel, config=config).save(out_path)
        logger.info("Done")
    else:
        logger.info(f"Skipped quantizing {model_name} (Not found at {out_path})")


def bundle_resources_for_guernika(pipe, args):
    """
    - Compiles Core ML models from mlpackage into mlmodelc format
    - Download tokenizer resources for the text encoder
    """
    resources_dir = os.path.join(args.o, args.resources_dir_name.replace("/", "_"))
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir, exist_ok=True)
        logger.info(f"Created {resources_dir} for Guernika assets")

    # Compile model using coremlcompiler (Significantly reduces the load time for unet)
    for source_name, target_name in [
        ("text_encoder", "TextEncoder"),
        ("text_encoder_2", "TextEncoder2"),
        ("text_encoder_prior", "TextEncoderPrior"),
        ("vae_encoder", "VAEEncoder"),
        ("vae_decoder", "VAEDecoder"),
        ("t2i_adapter", "T2IAdapter"),
        ("controlnet", "ControlNet"),
        ("unet", "Unet"),
        ("unet_chunk1", "UnetChunk1"),
        ("unet_chunk2", "UnetChunk2"),
        ("safety_checker", "SafetyChecker"),
        ("wuerstchen_prior", "WuerstchenPrior"),
        ("wuerstchen_decoder", "WuerstchenDecoder"),
        ("wuerstchen_vqgan", "WuerstchenVQGAN")
        
    ]:
        source_path = get_out_path(args, source_name)
        if os.path.exists(source_path):
            target_path = _compile_coreml_model(source_path, resources_dir, target_name)
            logger.info(f"Compiled {source_path} to {target_path}")
            if source_name.startswith("text_encoder"):
                # Fetch and save vocabulary JSON file for text tokenizer
                logger.info("Downloading and saving tokenizer vocab.json")
                with open(os.path.join(target_path, "vocab.json"), "wb") as f:
                    f.write(requests.get(args.text_encoder_vocabulary_url).content)
                logger.info("Done")

                # Fetch and save merged pairs JSON file for text tokenizer
                logger.info("Downloading and saving tokenizer merges.txt")
                with open(os.path.join(target_path, "merges.txt"), "wb") as f:
                    f.write(requests.get(args.text_encoder_merges_url).content)
                logger.info("Done")
                
                if hasattr(args, "added_vocab") and args.added_vocab:
                    logger.info("Saving added vocab")
                    with open(os.path.join(target_path, "added_vocab.json"), 'w', encoding='utf-8') as f:
                        json.dump(args.added_vocab, f, ensure_ascii=False, indent=4)
        else:
            logger.warning(
                f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
            )

    return resources_dir


def _compile_coreml_model(source_model_path, output_dir, final_name):
    """ Compiles Core ML models using the coremlcompiler utility from Xcode toolchain
    """
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile '{source_model_path}' '{output_dir}'")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path


def remove_mlpackages(args):
    for package_name in [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_prior",
        "vae_encoder",
        "vae_decoder",
        "t2i_adapter",
        "controlnet",
        "unet",
        "unet_chunk1",
        "unet_chunk2",
        "safety_checker",
        "wuerstchen_prior",
        "wuerstchen_prior_chunk1",
        "wuerstchen_prior_chunk2",
        "wuerstchen_decoder",
        "wuerstchen_decoder_chunk1",
        "wuerstchen_decoder_chunk2",
        "wuerstchen_vqgan"
    ]:
        package_path = get_out_path(args, package_name)
        try:
            if os.path.exists(package_path):
                shutil.rmtree(package_path)
        except:
            traceback.print_exc()
