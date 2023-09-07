#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools import (
    unet, controlnet, chunk_mlprogram
)
from guernikatools.merge_lora import merge_to_sd_model

import json
import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionInpaintPipeline
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

from types import MethodType


def _get_coreml_inputs(sample_inputs, samples_shapes, args):
    return [
        ct.TensorType(
            name=k,
            shape=samples_shapes[k] if samples_shapes and k in samples_shapes else v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


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
        raise ValueError(f"{final_psnr:.1f} dB is too low!")
    else:
        logger.info(
            f"{final_psnr:.1f} dB > {ABSOLUTE_MIN_PSNR} dB (minimum allowed) parity check passed"
        )
    return final_psnr


def _get_out_path(args, submodule_name):
    fname = f"{args.model_version}_{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    if args.clean_up_mlpackages:
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, fname)
    return os.path.join(args.o, fname)


def _convert_to_coreml(submodule_name, torchscript_module, coreml_inputs,
                       output_names, precision_full, args):
    out_path = _get_out_path(args, submodule_name)

    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        logger.info(f"Loading model from {out_path}")

        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(
            out_path, compute_units=ct.ComputeUnit[args.compute_unit])
        logger.info(
            f"Loading {out_path} took {time.time() - start:.1f} seconds")

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

        coreml_model.save(out_path)
        logger.info(f"Saved {submodule_name} model to {out_path}")

    return coreml_model, out_path
    
                
def quantize_weights(args):
    """ Quantize weights to args.quantize_nbits using a palette (look-up table)
    """
    models_to_quantize = ["text_encoder", "text_encoder_2", "vae_encoder", "vae_decoder", "unet", "unet_chunk1", "unet_chunk2", "controlnet"]
    if args.chunk_unet:
        models_to_quantize = ["text_encoder", "text_encoder_2", "vae_encoder", "vae_decoder", "unet_chunk1", "unet_chunk2", "controlnet"]
    for model_name in models_to_quantize:
        logger.info(f"Quantizing {model_name} to {args.quantize_nbits}-bit precision")
        out_path = _get_out_path(args, model_name)
        _quantize_weights(
            out_path,
            model_name,
            args.quantize_nbits
        )

def _quantize_weights(out_path, model_name, nbits):
    if os.path.exists(out_path):
        logger.info(f"Quantizing {model_name}")
        mlmodel = ct.models.MLModel(out_path,
                                    compute_units=ct.ComputeUnit.CPU_ONLY)

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
        logger.info(
            f"Skipped quantizing {model_name} (Not found at {out_path})")


def _compile_coreml_model(source_model_path, output_dir, final_name):
    """ Compiles Core ML models using the coremlcompiler utility from Xcode toolchain
    """
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(
            f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(
        os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile '{source_model_path}' '{output_dir}'")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path

def controlnet_method_from(identifier):
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
    for source_name, target_name in [("text_encoder", "TextEncoder"),
                                     ("text_encoder_2", "TextEncoder2"),
                                     ("vae_encoder", "VAEEncoder"),
                                     ("vae_decoder", "VAEDecoder"),
                                     ("controlnet", "ControlNet"),
                                     ("unet", "Unet"),
                                     ("unet_chunk1", "UnetChunk1"),
                                     ("unet_chunk2", "UnetChunk2"),
                                     ("safety_checker", "SafetyChecker")]:
        source_path = _get_out_path(args, source_name)
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

import traceback
def remove_mlpackages(args):
    for package_name in ["text_encoder", "text_encoder_2", "vae_encoder", "vae_decoder", "controlnet", "unet", "unet_chunk1", "unet_chunk2", "safety_checker"]:
        package_path = _get_out_path(args, package_name)
        try:
            if os.path.exists(package_path):
                shutil.rmtree(package_path)
        except:
            traceback.print_exc()

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

def convert_text_encoder(pipe, args):
    """ Converts the text encoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "text_encoder")
    if os.path.exists(out_path):
        logger.info(
            f"`text_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = pipe.tokenizer.model_max_length
    text_encoder_hidden_size = pipe.text_encoder.config.hidden_size

    sample_text_encoder_inputs = {
        "input_ids":
        torch.randint(
            pipe.text_encoder.config.vocab_size,
            (1, text_encoder_sequence_length),
            # https://github.com/apple/coremltools/issues/1423
            dtype=torch.float32,
        )
    }
    sample_text_encoder_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_text_encoder_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_text_encoder_inputs_spec}")

    class TextEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.text_encoder = pipe.text_encoder

        def forward(self, input_ids):
            return self.text_encoder(input_ids, return_dict=False)

    class TextEncoderXL(nn.Module):

        def __init__(self):
            super().__init__()
            self.text_encoder = pipe.text_encoder

        def forward(self, input_ids):
            output = self.text_encoder(input_ids, output_hidden_states=True)
            return (output.hidden_states[-2], output[0])
    
    reference_text_encoder = TextEncoderXL().eval() if args.model_is_sdxl else TextEncoder().eval()

    logger.info("JIT tracing text_encoder..")
    reference_text_encoder = torch.jit.trace(
        reference_text_encoder,
        (sample_text_encoder_inputs["input_ids"].to(torch.int32), ),
    )
    logger.info("Done.")

    sample_coreml_inputs = _get_coreml_inputs(sample_text_encoder_inputs, None, args)
    coreml_text_encoder, out_path = _convert_to_coreml(
        "text_encoder", reference_text_encoder, sample_coreml_inputs,
        ["last_hidden_state", "pooled_outputs"], args.precision_full, args)

    # Set model metadata
    coreml_text_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_text_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_text_encoder.version = args.model_version
    coreml_text_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_text_encoder.input_description["input_ids"] = "The token ids that represent the input text"

    # Set the output descriptions
    coreml_text_encoder.output_description["last_hidden_state"] = "The token embeddings as encoded by the Transformer model"
    coreml_text_encoder.output_description["pooled_outputs"] = "The version of the `last_hidden_state` output after pooling"
    
    # Set package version metadata
    coreml_text_encoder.user_defined_metadata["identifier"] = args.model_version
    coreml_text_encoder.user_defined_metadata["converter_version"] = __version__
    coreml_text_encoder.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_text_encoder.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_text_encoder.user_defined_metadata["hidden_size"] = str(pipe.text_encoder.config.hidden_size)

    coreml_text_encoder.save(out_path)

    logger.info(f"Saved text_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = pipe.text_encoder(
            sample_text_encoder_inputs["input_ids"].to(torch.int32),
            return_dict=False,
        )[1].numpy()

        coreml_out = list(
            coreml_text_encoder.predict(
                {k: v.numpy()
                 for k, v in sample_text_encoder_inputs.items()}).values())[0]
        report_correctness(
            baseline_out, coreml_out,
            "text_encoder baseline PyTorch to reference CoreML")

    del reference_text_encoder, coreml_text_encoder, pipe.text_encoder
    gc.collect()

def convert_text_encoder_2(pipe, args):
    """ Converts the text encoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "text_encoder_2")
    if os.path.exists(out_path):
        logger.info(
            f"`text_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = pipe.tokenizer_2.model_max_length
    text_encoder_hidden_size = pipe.text_encoder_2.config.hidden_size

    sample_text_encoder_inputs = {
        "input_ids":
        torch.randint(
            pipe.text_encoder_2.config.vocab_size,
            (1, text_encoder_sequence_length),
            # https://github.com/apple/coremltools/issues/1423
            dtype=torch.float32,
        )
    }
    sample_text_encoder_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_text_encoder_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_text_encoder_inputs_spec}")

    class TextEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.text_encoder = pipe.text_encoder_2

        def forward(self, input_ids):
            output = self.text_encoder(input_ids, output_hidden_states=True)
            return (output.hidden_states[-2], output[0])

    reference_text_encoder = TextEncoder().eval()

    logger.info("JIT tracing text_encoder_2..")
    reference_text_encoder = torch.jit.trace(
        reference_text_encoder,
        (sample_text_encoder_inputs["input_ids"].to(torch.int32), ),
    )
    logger.info("Done.")

    sample_coreml_inputs = _get_coreml_inputs(sample_text_encoder_inputs, None, args)
    coreml_text_encoder, out_path = _convert_to_coreml(
        "text_encoder_2", reference_text_encoder, sample_coreml_inputs,
        ["last_hidden_state", "pooled_outputs"], args.precision_full, args)

    # Set model metadata
    coreml_text_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_text_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_text_encoder.version = args.model_version
    coreml_text_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_text_encoder.input_description["input_ids"] = "The token ids that represent the input text"

    # Set the output descriptions
    coreml_text_encoder.output_description["last_hidden_state"] = "The token embeddings as encoded by the Transformer model"
    coreml_text_encoder.output_description["pooled_outputs"] = "The version of the `last_hidden_state` output after pooling"
    
    # Set package version metadata
    coreml_text_encoder.user_defined_metadata["identifier"] = args.model_version
    coreml_text_encoder.user_defined_metadata["converter_version"] = __version__
    coreml_text_encoder.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_text_encoder.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_text_encoder.user_defined_metadata["hidden_size"] = str(pipe.text_encoder_2.config.hidden_size)

    coreml_text_encoder.save(out_path)

    logger.info(f"Saved text_encoder_2 into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = pipe.text_encoder_2(
            sample_text_encoder_inputs["input_ids"].to(torch.int32),
            return_dict=False,
        )[1].numpy()

        coreml_out = list(
            coreml_text_encoder.predict(
                {k: v.numpy()
                 for k, v in sample_text_encoder_inputs.items()}).values())[0]
        report_correctness(
            baseline_out, coreml_out,
            "text_encoder baseline PyTorch to reference CoreML")

    del reference_text_encoder, coreml_text_encoder, pipe.text_encoder_2
    gc.collect()

def modify_coremltools_torch_frontend_badbmm():
    """
    Modifies coremltools torch frontend for baddbmm to be robust to the `beta` argument being of non-float dtype:
    e.g. https://github.com/huggingface/diffusers/blob/v0.8.1/src/diffusers/models/attention.py#L315
    """
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    if "baddbmm" in _TORCH_OPS_REGISTRY:
        del _TORCH_OPS_REGISTRY["baddbmm"]

    @register_torch_op
    def baddbmm(context, node):
        """
        baddbmm(Tensor input, Tensor batch1, Tensor batch2, Scalar beta=1, Scalar alpha=1)
        output = beta * input + alpha * batch1 * batch2
        Notice that batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, then input must be broadcastable with a (b×n×p) tensor
        and out will be a (b×n×p) tensor.
        """
        assert len(node.outputs) == 1
        inputs = _get_inputs(context, node, expected=5)
        bias, batch1, batch2, beta, alpha = inputs

        if beta.val != 1.0:
            # Apply scaling factor beta to the bias.
            if beta.val.dtype == np.int32:
                beta = mb.cast(x=beta, dtype="fp32")
                logger.warning(
                    f"Casted the `beta`(value={beta.val}) argument of `baddbmm` op "
                    "from int32 to float32 dtype for conversion!")
            bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")

            context.add(bias)

        if alpha.val != 1.0:
            # Apply scaling factor alpha to the input.
            batch1 = mb.mul(x=alpha, y=batch1, name=batch1.name + "_scaled")
            context.add(batch1)

        bmm_node = mb.matmul(x=batch1, y=batch2, name=node.name + "_bmm")
        context.add(bmm_node)

        baddbmm_node = mb.add(x=bias, y=bmm_node, name=node.name)
        context.add(baddbmm_node)


def convert_vae_encoder(pipe, args):
    """ Converts the VAE Encoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "vae_encoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_encoder() before convert_unet()")
    
    z_shape = (
        1,  # B
        3,  # C
        args.output_h,  # H
        args.output_w,  # w
    )

    sample_vae_encoder_inputs = {
        "z": torch.rand(*z_shape, dtype=torch.float16)
    }

    class VAEEncoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.encode() method
        """

        def __init__(self):
            super().__init__()
            self.quant_conv = pipe.vae.quant_conv
            self.encoder = pipe.vae.encoder

        def forward(self, z):
            return self.quant_conv(self.encoder(z))

    baseline_encoder = VAEEncoder().eval()

    # No optimization needed for the VAE Encoder as it is a pure ConvNet
    traced_vae_encoder = torch.jit.trace(baseline_encoder, (sample_vae_encoder_inputs["z"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
            
    if args.multisize:
        sample_size = args.output_h
        input_shape = ct.Shape(shape=(
            1,
            3,
            ct.RangeDim(lower_bound=int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
            ct.RangeDim(lower_bound=int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size)
        ))
        sample_coreml_inputs = _get_coreml_inputs(sample_vae_encoder_inputs, {"z": input_shape}, args)
    else:
        sample_coreml_inputs = _get_coreml_inputs(sample_vae_encoder_inputs, None, args)
    
    # SDXL seems to require full precision
    precision_full = args.model_is_sdxl and args.model_version != "stabilityai/stable-diffusion-xl-base-1.0"
    coreml_vae_encoder, out_path = _convert_to_coreml(
        "vae_encoder", traced_vae_encoder, sample_coreml_inputs,
        ["latent"], precision_full, args)

    # Set model metadata
    coreml_vae_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_vae_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_vae_encoder.version = args.model_version
    coreml_vae_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_vae_encoder.input_description["z"] = \
        "The input image to base the initial latents on normalized to range [-1, 1]"

    # Set the output descriptions
    coreml_vae_encoder.output_description["latent"] = "The latent embeddings from the unet model from the input image."
    
    # Set package version metadata
    coreml_vae_encoder.user_defined_metadata["identifier"] = args.model_version
    coreml_vae_encoder.user_defined_metadata["converter_version"] = __version__
    coreml_vae_encoder.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_vae_encoder.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_vae_encoder.user_defined_metadata["scaling_factor"] = str(pipe.vae.config.scaling_factor)

    coreml_vae_encoder.save(out_path)

    logger.info(f"Saved vae_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_encoder(
            z=sample_vae_encoder_inputs["z"].to(torch.float32)).numpy()
        coreml_out = list(
            coreml_vae_encoder.predict(
                {k: v.numpy()
                 for k, v in sample_vae_encoder_inputs.items()}).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "vae_encoder baseline PyTorch to baseline CoreML")

    del traced_vae_encoder, pipe.vae.encoder, coreml_vae_encoder
    gc.collect()


def convert_vae_decoder(pipe, args):
    """ Converts the VAE Decoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "vae_decoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_decoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_decoder() before convert_unet()")
    
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    height = int(args.output_h / vae_scale_factor)
    width = int(args.output_w / vae_scale_factor)
    vae_latent_channels = pipe.vae.config.latent_channels
    z_shape = (
        1,  # B
        vae_latent_channels,  # C
        height,  # H
        width,  # w
    )

    sample_vae_decoder_inputs = {
        "z": torch.rand(*z_shape, dtype=torch.float16)
    }

    class VAEDecoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.decode() method
        """

        def __init__(self):
            super().__init__()
            self.post_quant_conv = pipe.vae.post_quant_conv
            self.decoder = pipe.vae.decoder

        def forward(self, z):
            return self.decoder(self.post_quant_conv(z))

    baseline_decoder = VAEDecoder().eval()

    # No optimization needed for the VAE Decoder as it is a pure ConvNet
    traced_vae_decoder = torch.jit.trace(baseline_decoder, (sample_vae_decoder_inputs["z"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
        
    if args.multisize:
        sample_size = height
        input_shape = ct.Shape(shape=(
            1,
            vae_latent_channels,
            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size)
        ))
        sample_coreml_inputs = _get_coreml_inputs(sample_vae_decoder_inputs, {"z": input_shape}, args)
    else:
        sample_coreml_inputs = _get_coreml_inputs(sample_vae_decoder_inputs, None, args)
    
    # SDXL seems to require full precision
    precision_full = args.model_is_sdxl and args.model_version != "stabilityai/stable-diffusion-xl-base-1.0"
    coreml_vae_decoder, out_path = _convert_to_coreml(
        "vae_decoder", traced_vae_decoder, sample_coreml_inputs,
        ["image"], precision_full, args)

    # Set model metadata
    coreml_vae_decoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_vae_decoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_vae_decoder.version = args.model_version
    coreml_vae_decoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_vae_decoder.input_description["z"] = \
        "The denoised latent embeddings from the unet model after the last step of reverse diffusion"

    # Set the output descriptions
    coreml_vae_decoder.output_description["image"] = "Generated image normalized to range [-1, 1]"
    
    # Set package version metadata
    coreml_vae_decoder.user_defined_metadata["identifier"] = args.model_version
    coreml_vae_decoder.user_defined_metadata["converter_version"] = __version__
    coreml_vae_decoder.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_vae_decoder.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_vae_decoder.user_defined_metadata["scaling_factor"] = str(pipe.vae.config.scaling_factor)

    coreml_vae_decoder.save(out_path)

    logger.info(f"Saved vae_decoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_decoder(z=sample_vae_decoder_inputs["z"].to(torch.float32)).numpy()
        coreml_out = list(coreml_vae_decoder.predict({
                k: v.numpy() for k, v in sample_vae_decoder_inputs.items()
        }).values())[0]
        report_correctness(baseline_out, coreml_out, "vae_decoder baseline PyTorch to baseline CoreML")

    del traced_vae_decoder, pipe.vae.decoder, coreml_vae_decoder
    gc.collect()


def convert_controlnet(pipe, args):
    """ Converts the ControlNet component of Stable Diffusion
    """
    if not pipe.controlnet:
        logger.info(
            f"`controlnet` not available in this pipline."
        )
        return
    
    out_path = _get_out_path(args, "controlnet")
    if os.path.exists(out_path):
        logger.info(
            f"`controlnet` already exists at {out_path}, skipping conversion."
        )
        return

    # Register the selected attention implementation globally
    unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations[args.attention_implementation]
    logger.info(
        f"Attention implementation in effect: {unet.ATTENTION_IMPLEMENTATION_IN_EFFECT}"
    )

    # Prepare sample input shapes and values
    batch_size = 2  # for classifier-free guidance
    controlnet_in_channels = pipe.controlnet.config.in_channels
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    height = int(args.output_h / vae_scale_factor)
    width = int(args.output_w / vae_scale_factor)
    
    # if using variable size shapes, take the biggest as base
    if args.multisize and height != width:
        height = max(height, width)
        width = height
    
    sample_shape = (
        batch_size,                    # B
        controlnet_in_channels,  # C
        height,  # H
        width,  # W
    )
    
    cond_shape = (
        batch_size,                    # B
        3,  # C
        args.output_h,  # H
        args.output_w,  # W
    )

    if not hasattr(pipe, "text_encoder"):
        raise RuntimeError(
            "convert_text_encoder() deletes pipe.text_encoder to save RAM. "
            "Please use convert_controlnet() before convert_text_encoder()")

    hidden_size = pipe.controlnet.config.cross_attention_dim
    encoder_hidden_states_shape = (
        batch_size,
        hidden_size,
        1,
        pipe.text_encoder.config.max_position_embeddings,
    )

    # Create the scheduled timesteps for downstream use
    DEFAULT_NUM_INFERENCE_STEPS = 50
    pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)
    
    output_names = [
        "down_block_res_samples_00", "down_block_res_samples_01", "down_block_res_samples_02",
        "down_block_res_samples_03", "down_block_res_samples_04", "down_block_res_samples_05",
        "down_block_res_samples_06", "down_block_res_samples_07", "down_block_res_samples_08"
    ]
    sample_controlnet_inputs = [
        ("sample", torch.rand(*sample_shape)),
        ("timestep",
         torch.tensor([pipe.scheduler.timesteps[0].item()] *
                      (batch_size)).to(torch.float32)),
        ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ("controlnet_cond", torch.rand(*cond_shape))
    ]
    if hasattr(pipe.controlnet.config, "addition_embed_type") and pipe.controlnet.config.addition_embed_type == "text_time":
        text_embeds_shape = (
            batch_size,
            pipe.text_encoder_2.config.hidden_size,
        )
        time_ids_input = [
            [args.output_h, args.output_w, 0, 0, args.output_h, args.output_w],
            [args.output_h, args.output_w, 0, 0, args.output_h, args.output_w]
        ]
        sample_controlnet_inputs = sample_controlnet_inputs + [
            ("text_embeds", torch.rand(*text_embeds_shape)),
            ("time_ids", torch.tensor(time_ids_input).to(torch.float32)),
        ]
    else:
        # SDXL ControlNet does not generate these outputs
        output_names = output_names + ["down_block_res_samples_09", "down_block_res_samples_10", "down_block_res_samples_11"]
    output_names = output_names + ["mid_block_res_sample"]

    sample_controlnet_inputs = OrderedDict(sample_controlnet_inputs)
    sample_controlnet_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_controlnet_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_controlnet_inputs_spec}")

    # Initialize reference controlnet
    reference_controlnet = controlnet.ControlNetModel(**pipe.controlnet.config).eval()
    load_state_dict_summary = reference_controlnet.load_state_dict(pipe.controlnet.state_dict())

    # Prepare inputs
    baseline_sample_controlnet_inputs = deepcopy(sample_controlnet_inputs)
    baseline_sample_controlnet_inputs[
        "encoder_hidden_states"] = baseline_sample_controlnet_inputs[
            "encoder_hidden_states"].squeeze(2).transpose(1, 2)

    # JIT trace
    logger.info("JIT tracing..")
    reference_controlnet = torch.jit.trace(reference_controlnet, example_kwarg_inputs=sample_controlnet_inputs)
    logger.info("Done.")

    if args.check_output_correctness:
        baseline_out = pipe.controlnet(**baseline_sample_controlnet_inputs, return_dict=False)[0].numpy()
        reference_out = reference_controlnet(**sample_controlnet_inputs)[0].numpy()
        report_correctness(baseline_out, reference_out,  "control baseline to reference PyTorch")

    del pipe.controlnet
    gc.collect()

    coreml_sample_controlnet_inputs = {
        k: v.numpy().astype(np.float16)
        for k, v in sample_controlnet_inputs.items()
    }
    
    if args.multisize:
        sample_size = height
        sample_input_shape = ct.Shape(shape=(
            batch_size,
            controlnet_in_channels,
            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size)
        ))
        
        cond_size = args.output_h
        cond_input_shape = ct.Shape(shape=(
            batch_size,
            3,
            ct.RangeDim(int(cond_size * 0.5), upper_bound=int(cond_size * 2), default=cond_size),
            ct.RangeDim(int(cond_size * 0.5), upper_bound=int(cond_size * 2), default=cond_size)
        ))
        
        sample_coreml_inputs = _get_coreml_inputs(coreml_sample_controlnet_inputs, {
            "sample": sample_input_shape,
            "controlnet_cond": cond_input_shape,
        }, args)
    else:
        sample_coreml_inputs = _get_coreml_inputs(coreml_sample_controlnet_inputs, None, args)
    coreml_controlnet, out_path = _convert_to_coreml(
        "controlnet",
        reference_controlnet,
        sample_coreml_inputs,
        output_names,
        args.precision_full,
        args
    )
    del reference_controlnet
    gc.collect()

    # Set model metadata
    coreml_controlnet.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_controlnet.license = "ControlNet (https://github.com/lllyasviel/ControlNet)"
    coreml_controlnet.version = args.controlnet_version
    coreml_controlnet.short_description = \
        "ControlNet is a neural network structure to control diffusion models by adding extra conditions. " \
        "Please refer to https://github.com/lllyasviel/ControlNet for details."

    # Set the input descriptions
    coreml_controlnet.input_description["sample"] = \
        "The low resolution latent feature maps being denoised through reverse diffusion"
    coreml_controlnet.input_description["timestep"] = \
        "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
    coreml_controlnet.input_description["encoder_hidden_states"] = \
        "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
        "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. " \
        "Shorter text does not reduce computation."
    coreml_controlnet.input_description["controlnet_cond"] = \
        "Image used to condition ControlNet output"

    # Set the output descriptions
    coreml_controlnet.output_description["down_block_res_samples_01"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_02"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_03"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_04"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_05"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_06"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_07"] = \
        "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_08"] = \
        "Residual down sample from ControlNet"
    if "down_block_res_samples_09" in output_names:
        coreml_controlnet.output_description["down_block_res_samples_09"] = \
            "Residual down sample from ControlNet"
        coreml_controlnet.output_description["down_block_res_samples_10"] = \
            "Residual down sample from ControlNet"
        coreml_controlnet.output_description["down_block_res_samples_11"] = \
            "Residual down sample from ControlNet"
    coreml_controlnet.output_description["mid_block_res_sample"] = \
        "Residual mid sample from ControlNet"
    
    # Set package version metadata
    coreml_controlnet.user_defined_metadata["identifier"] = args.controlnet_version
    coreml_controlnet.user_defined_metadata["converter_version"] = __version__
    coreml_controlnet.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_controlnet.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_controlnet.user_defined_metadata["hidden_size"] = str(hidden_size)
    controlnet_method = controlnet_method_from(args.controlnet_version)
    if controlnet_method:
        coreml_controlnet.user_defined_metadata["method"] = controlnet_method
    
    coreml_controlnet.save(out_path)
    logger.info(f"Saved controlnet into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        coreml_out = list(
            coreml_controlnet.predict(coreml_sample_controlnet_inputs).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "control baseline PyTorch to reference CoreML")

    del coreml_controlnet
    gc.collect()


def convert_unet(pipe, args):
    """ Converts the UNet component of Stable Diffusion
    """
    out_path = _get_out_path(args, "unet")

    # Check if Unet was previously exported and then chunked
    unet_chunks_exist = all(
        os.path.exists(out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage")) for idx in range(2)
    )

    if args.chunk_unet and unet_chunks_exist:
        logger.info("`unet` chunks already exist, skipping conversion.")
        del pipe.unet
        gc.collect()
        return

    # If original Unet does not exist, export it from PyTorch+diffusers
    elif not os.path.exists(out_path):
        # Register the selected attention implementation globally
        unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations[args.attention_implementation]
        logger.info(f"Attention implementation in effect: {unet.ATTENTION_IMPLEMENTATION_IN_EFFECT}")

        # Prepare sample input shapes and values
        batch_size = 2  # for classifier-free guidance
        unet_in_channels = pipe.unet.config.in_channels
        # allow converting instruct pix2pix
        if unet_in_channels == 8:
            batch_size = 3
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = int(args.output_h / vae_scale_factor)
        width = int(args.output_w / vae_scale_factor)
        
        # if using variable size shapes, take the biggest as base
        if args.multisize and height != width:
            height = max(height, width)
            width = height
        
        sample_shape = (
            batch_size,                    # B
            unet_in_channels,  # C
            height,  # H
            width,  # W
        )
        
        max_position_embeddings = 77
        if hasattr(pipe, "text_encoder") and pipe.text_encoder and pipe.text_encoder.config:
            max_position_embeddings = pipe.text_encoder.config.max_position_embeddings
        elif hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 and pipe.text_encoder_2.config:
            max_position_embeddings = pipe.text_encoder_2.config.max_position_embeddings
        else:
            raise RuntimeError(
                "convert_text_encoder() deletes pipe.text_encoder to save RAM. "
                "Please use convert_unet() before convert_text_encoder()")

        hidden_size = pipe.unet.config.cross_attention_dim
        encoder_hidden_states_shape = (
            batch_size,
            hidden_size,
            1,
            max_position_embeddings,
        )

        # Create the scheduled timesteps for downstream use
        DEFAULT_NUM_INFERENCE_STEPS = 50
        pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)
        
        sample_unet_inputs = [
            ("sample", torch.rand(*sample_shape)),
            ("timestep", torch.tensor(
                [pipe.scheduler.timesteps[0].item()] * (batch_size)
            ).to(torch.float32)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ]
        
        requires_aesthetics_score = False
        if hasattr(pipe.unet.config, "addition_embed_type") and pipe.unet.config.addition_embed_type == "text_time":
            text_embeds_shape = (
                batch_size,
                pipe.text_encoder_2.config.hidden_size,
            )
            time_ids_input = None
            if hasattr(pipe.config, "requires_aesthetics_score") and pipe.config.requires_aesthetics_score:
                requires_aesthetics_score = True
                time_ids_input = [
                    [args.output_h, args.output_w, 0, 0, 2.5],
                    [args.output_h, args.output_w, 0, 0, 6]
                ]
            else:
                time_ids_input = [
                    [args.output_h, args.output_w, 0, 0, args.output_h, args.output_w],
                    [args.output_h, args.output_w, 0, 0, args.output_h, args.output_w]
                ]
            sample_unet_inputs = sample_unet_inputs + [
                ("text_embeds", torch.rand(*text_embeds_shape)),
                ("time_ids", torch.tensor(time_ids_input).to(torch.float32)),
            ]
        
        if args.controlnet_support:
            block_out_channels = pipe.unet.config.block_out_channels
                                    
            cn_output = 0
            cn_height = height
            cn_width = width
            
            # down
            output_channel = block_out_channels[0]
            sample_unet_inputs = sample_unet_inputs + [
                (f"down_block_res_samples_{cn_output:02}", torch.rand(2, output_channel, cn_height, cn_width))
            ]
            cn_output += 1
            
            for i, output_channel in enumerate(block_out_channels):
                is_final_block = i == len(block_out_channels) - 1
                sample_unet_inputs = sample_unet_inputs + [
                    (f"down_block_res_samples_{cn_output:02}", torch.rand(2,  output_channel, cn_height, cn_width)),
                    (f"down_block_res_samples_{cn_output+1:02}", torch.rand(2,  output_channel, cn_height, cn_width)),
                ]
                cn_output += 2
                if not is_final_block:
                    cn_height = int(cn_height / 2)
                    cn_width = int(cn_width / 2)
                    sample_unet_inputs = sample_unet_inputs + [
                        (f"down_block_res_samples_{cn_output:02}", torch.rand(2,  output_channel, cn_height, cn_width)),
                    ]
                    cn_output += 1
            
            # mid
            output_channel = block_out_channels[-1]
            sample_unet_inputs = sample_unet_inputs + [
                ("mid_block_res_sample", torch.rand(2, output_channel, cn_height, cn_width))
            ]
        
        multisize_inputs = None
        if args.multisize:
            sample_size = height
            input_shape = ct.Shape(shape=(
                batch_size,
                unet_in_channels,
                ct.RangeDim(lower_bound=int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
                ct.RangeDim(lower_bound=int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size)
            ))
            multisize_inputs = {"sample": input_shape}
            for k, v in sample_unet_inputs:
                if "block_res" in k:
                    v_height = v.shape[2]
                    v_width = v.shape[3]
                    multisize_inputs[k] = ct.Shape(shape=(
                        2,
                        output_channel,
                        ct.RangeDim(lower_bound=int(v_height * 0.5), upper_bound=int(v_height * 2), default=v_height),
                        ct.RangeDim(lower_bound=int(v_width * 0.5), upper_bound=int(v_width * 2), default=v_width)
                    ))
        
        sample_unet_inputs = OrderedDict(sample_unet_inputs)
        sample_unet_inputs_spec = {
            k: (v.shape, v.dtype)
            for k, v in sample_unet_inputs.items()
        }
        logger.info(f"Sample inputs spec: {sample_unet_inputs_spec}")

        # Initialize reference unet
        reference_unet = unet.UNet2DConditionModel(**pipe.unet.config).eval()
        load_state_dict_summary = reference_unet.load_state_dict(pipe.unet.state_dict())

        # Prepare inputs
        baseline_sample_unet_inputs = deepcopy(sample_unet_inputs)
        baseline_sample_unet_inputs[
            "encoder_hidden_states"] = baseline_sample_unet_inputs[
                "encoder_hidden_states"].squeeze(2).transpose(1, 2)
        
        if not args.check_output_correctness:
            del pipe.unet
            gc.collect()
        
        # JIT trace
        logger.info("JIT tracing..")
        reference_unet = torch.jit.trace(reference_unet, example_kwarg_inputs=sample_unet_inputs)
        logger.info("Done.")

        if args.check_output_correctness:
            baseline_out = pipe.unet(**baseline_sample_unet_inputs, return_dict=False)[0].detach().numpy()
            reference_out = reference_unet(**sample_unet_inputs)[0].detach().numpy()
            report_correctness(baseline_out, reference_out, "unet baseline to reference PyTorch")

            del pipe.unet
            gc.collect()

        coreml_sample_unet_inputs = {
            k: v.numpy().astype(np.float16)
            for k, v in sample_unet_inputs.items()
        }
        
        sample_coreml_inputs = _get_coreml_inputs(coreml_sample_unet_inputs, multisize_inputs, args)
        precision_full = args.precision_full
        if not precision_full and pipe.scheduler.config.prediction_type == "v_prediction":
            precision_full = True
            logger.info(f"Full precision required: prediction_type == v_prediction")
        coreml_unet, out_path = _convert_to_coreml("unet", reference_unet,
                                                   sample_coreml_inputs,
                                                   ["noise_pred"], precision_full, args)
        del reference_unet
        gc.collect()

        # Set model metadata
        coreml_unet.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
        coreml_unet.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
        coreml_unet.version = args.model_version
        coreml_unet.short_description = \
            "Stable Diffusion generates images conditioned on text or other images as input through the diffusion process. " \
            "Please refer to https://arxiv.org/abs/2112.10752 for details."

        # Set the input descriptions
        coreml_unet.input_description["sample"] = \
            "The low resolution latent feature maps being denoised through reverse diffusion"
        coreml_unet.input_description["timestep"] = \
            "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
        coreml_unet.input_description["encoder_hidden_states"] = \
            "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
            "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. " \
            "Shorter text does not reduce computation."
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2:
            coreml_unet.input_description["text_embeds"] = \
                ""
            coreml_unet.input_description["time_ids"] = \
                ""
        if args.controlnet_support:
            coreml_unet.input_description["down_block_res_samples_00"] = \
                "Optional: Residual down sample from ControlNet"
            coreml_unet.input_description["mid_block_res_sample"] = \
                "Optional: Residual mid sample from ControlNet"

        # Set the output descriptions
        coreml_unet.output_description["noise_pred"] = \
            "Same shape and dtype as the `sample` input. " \
            "The predicted noise to facilitate the reverse diffusion (denoising) process"
            
        # Set package version metadata
        coreml_unet.user_defined_metadata["identifier"] = args.model_version
        coreml_unet.user_defined_metadata["converter_version"] = __version__
        coreml_unet.user_defined_metadata["attention_implementation"] = args.attention_implementation
        coreml_unet.user_defined_metadata["compute_unit"] = args.compute_unit
        coreml_unet.user_defined_metadata["prediction_type"] = pipe.scheduler.config.prediction_type
        coreml_unet.user_defined_metadata["hidden_size"] = str(hidden_size)
        if requires_aesthetics_score:
            coreml_unet.user_defined_metadata["requires_aesthetics_score"] = "true"
         
        coreml_unet.save(out_path)
        logger.info(f"Saved unet into {out_path}")

        # Parity check PyTorch vs CoreML
        if args.check_output_correctness:
            coreml_out = list(coreml_unet.predict(coreml_sample_unet_inputs).values())[0]
            report_correctness(baseline_out, coreml_out, "unet baseline PyTorch to reference CoreML")

        del coreml_unet
        gc.collect()
    else:
        del pipe.unet
        gc.collect()
        logger.info(f"`unet` already exists at {out_path}, skipping conversion.")

    if args.chunk_unet and not unet_chunks_exist:
        logger.info("Chunking unet in two approximately equal MLModels")
        args.mlpackage_path = out_path
        args.remove_original = False
        chunk_mlprogram.main(args)


def convert_safety_checker(pipe, args):
    """ Converts the Safety Checker component of Stable Diffusion
    """
    if pipe.safety_checker is None:
        logger.warning(
            f"diffusers pipeline for {args.model_version} does not have a `safety_checker` module! " \
            "`--convert-safety-checker` will be ignored."
        )
        return

    out_path = _get_out_path(args, "safety_checker")
    if os.path.exists(out_path):
        logger.info(
            f"`safety_checker` already exists at {out_path}, skipping conversion."
        )
        return

    sample_image = np.random.randn(
        1,  # B
        args.output_h,  # H
        args.output_w,  # w
        3  # C
    ).astype(np.float32)

    # Note that pipe.feature_extractor is not an ML model. It simply
    # preprocesses data for the pipe.safety_checker module.
    safety_checker_input = pipe.feature_extractor(
        pipe.numpy_to_pil(sample_image),
        return_tensors="pt",
    ).pixel_values.to(torch.float32)

    sample_safety_checker_inputs = OrderedDict([
        ("clip_input", safety_checker_input),
        ("images", torch.from_numpy(sample_image)),
        ("adjustment", torch.tensor([0]).to(torch.float32)),
    ])

    sample_safety_checker_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_safety_checker_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_safety_checker_inputs_spec}")

    # Patch safety_checker's forward pass to be vectorized and avoid conditional blocks
    # (similar to pipe.safety_checker.forward_onnx)
    from diffusers.pipelines.stable_diffusion import safety_checker

    def forward_coreml(self, clip_input, images, adjustment):
        """ Forward pass implementation for safety_checker
        """

        def cosine_distance(image_embeds, text_embeds):
            return F.normalize(image_embeds) @ F.normalize(text_embeds).transpose(0, 1)

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = special_scores.gt(0).float().sum(dim=1).gt(0).float()
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = concept_scores.gt(0).float().sum(dim=1).gt(0)
        
        # There is a problem when converting using multisize, for now the workaround is to not filter the images
        # The swift implementations already filters the images checking `has_nsfw_concepts` so this should not have any impact
        
        #has_nsfw_concepts = concept_scores.gt(0).float().sum(dim=1).gt(0)[:, None, None, None]

        #has_nsfw_concepts_inds, _ = torch.broadcast_tensors(has_nsfw_concepts, images)
        #images[has_nsfw_concepts_inds] = 0.0  # black image

        return images, has_nsfw_concepts.float(), concept_scores

    baseline_safety_checker = deepcopy(pipe.safety_checker.eval())
    setattr(baseline_safety_checker, "forward", MethodType(forward_coreml, baseline_safety_checker))

    # In order to parity check the actual signal, we need to override the forward pass to return `concept_scores` which is the
    # output before thresholding
    # Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L100
    def forward_extended_return(self, clip_input, images, adjustment):

        def cosine_distance(image_embeds, text_embeds):
            normalized_image_embeds = F.normalize(image_embeds)
            normalized_text_embeds = F.normalize(text_embeds)
            return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        # Don't make the images black as to align with the workaround in `forward_coreml`
        #images[has_nsfw_concepts] = 0.0

        return images, has_nsfw_concepts, concept_scores

    setattr(pipe.safety_checker, "forward", MethodType(forward_extended_return, pipe.safety_checker))

    # Trace the safety_checker model
    logger.info("JIT tracing..")
    traced_safety_checker = torch.jit.trace(baseline_safety_checker, list(sample_safety_checker_inputs.values()))
    logger.info("Done.")
    del baseline_safety_checker
    gc.collect()

    # Cast all inputs to float16
    coreml_sample_safety_checker_inputs = {
        k: v.numpy().astype(np.float16)
        for k, v in sample_safety_checker_inputs.items()
    }

    # Convert safety_checker model to Core ML
    if args.multisize:
        clip_size = safety_checker_input.shape[2]
        clip_input_shape = ct.Shape(shape=(
            1,
            3,
            ct.RangeDim(int(clip_size * 0.5), upper_bound=int(clip_size * 2), default=clip_size),
            ct.RangeDim(int(clip_size * 0.5), upper_bound=int(clip_size * 2), default=clip_size)
        ))
        sample_size = args.output_h
        input_shape = ct.Shape(shape=(
            1,
            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
            3
        ))
        
        sample_coreml_inputs = _get_coreml_inputs(coreml_sample_safety_checker_inputs, {
            "clip_input": clip_input_shape, "images": input_shape
        }, args)
    else:
        sample_coreml_inputs = _get_coreml_inputs(coreml_sample_safety_checker_inputs, None, args)
    coreml_safety_checker, out_path = _convert_to_coreml(
        "safety_checker", traced_safety_checker,
        sample_coreml_inputs,
        ["filtered_images", "has_nsfw_concepts", "concept_scores"], False, args)

    # Set model metadata
    coreml_safety_checker.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_safety_checker.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_safety_checker.version = args.model_version
    coreml_safety_checker.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_safety_checker.input_description["clip_input"] = \
        "The normalized image input tensor resized to (224x224) in channels-first (BCHW) format"
    coreml_safety_checker.input_description["images"] = \
        f"Output of the vae_decoder ({pipe.vae.config.sample_size}x{pipe.vae.config.sample_size}) in channels-last (BHWC) format"
    coreml_safety_checker.input_description["adjustment"] = \
        "Bias added to the concept scores to trade off increased recall for reduce precision in the safety checker classifier"

    # Set the output descriptions
    coreml_safety_checker.output_description["filtered_images"] = \
        f"Identical to the input `images`. If safety checker detected any sensitive content, " \
        "the corresponding image is replaced with a blank image (zeros)"
    coreml_safety_checker.output_description["has_nsfw_concepts"] = \
        "Indicates whether the safety checker model found any sensitive content in the given image"
    coreml_safety_checker.output_description["concept_scores"] = \
        "Concept scores are the scores before thresholding at zero yields the `has_nsfw_concepts` output. " \
        "These scores can be used to tune the `adjustment` input"
    
    # Set package version metadata
    coreml_safety_checker.user_defined_metadata["identifier"] = args.model_version
    coreml_safety_checker.user_defined_metadata["converter_version"] = __version__
    coreml_safety_checker.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_safety_checker.user_defined_metadata["compute_unit"] = args.compute_unit

    coreml_safety_checker.save(out_path)

    if args.check_output_correctness:
        baseline_out = pipe.safety_checker(**sample_safety_checker_inputs)[2].numpy()
        coreml_out = coreml_safety_checker.predict(coreml_sample_safety_checker_inputs)["concept_scores"]
        report_correctness(baseline_out, coreml_out, "safety_checker baseline PyTorch to reference CoreML")

    del traced_safety_checker, coreml_safety_checker, pipe.safety_checker
    gc.collect()

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
    
    logger.info(f"Output size will be {args.output_w}x{args.output_h}")

def main(args):
    os.makedirs(args.o, exist_ok=True)
    
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
            else:
                os.system(
                    f"curl https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml --output {temp_dir}/v1-inference.yaml"
                )
                original_config_file = f"{temp_dir}/v1-inference.yaml"
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
                pipe = StableDiffusionXLPipeline.from_single_file(args.model_checkpoint_location, local_files_only=True)
            except:
                try:
                    pipe = StableDiffusionPipeline.from_single_file(args.model_checkpoint_location, local_files_only=True)
                except:
                    pipe = StableDiffusionInpaintPipeline.from_single_file(args.model_checkpoint_location, local_files_only=True)
        else:
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=args.model_checkpoint_location,
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
        logger.info(f"Initializing StableDiffusionPipeline with {args.model_version}..")
        pipe = DiffusionPipeline.from_pretrained(args.model_version, use_auth_token=True)
        
    if args.embeddings_location:
        logger.info(f"Loading embeddings at {args.embeddings_location}")
        embeddings_files = [join(args.embeddings_location, f) for f in listdir(args.embeddings_location) if not f.startswith('.') and isfile(join(args.embeddings_location, f))]
        for file in embeddings_files:
            logger.info(f"Loading embedding: {file}")
            pipe.load_textual_inversion(file, local_files_only=True)
        args.added_vocab = pipe.tokenizer.get_added_vocab()
        logger.info(f"Added embeddings: {args.added_vocab}")
        
    args.multisize = False
    args.model_is_sdxl = hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2
    # auto apply fix if converting base SDXL
    if args.model_version == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
    
    if args.loras_to_merge:
        loras_locations = [lora_info.split(":", 1)[0] for lora_info in args.loras_to_merge]
        loras_ratios = [float(lora_info.split(":", 1)[1]) for lora_info in args.loras_to_merge]
        logger.info(f"Merging LoRAs at: {loras_locations}")
        
        if args.model_is_sdxl:
            merge_to_sd_model(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, loras_locations, loras_ratios)
        else:
            merge_to_sd_model(pipe.unet, pipe.text_encoder, None, loras_locations, loras_ratios)
    
    logger.info(f"Done.")
    check_output_size(pipe, args)
    
    # Convert models
    if controlnet:
        logger.info("Converting controlnet")
        convert_controlnet(pipe, args)
        logger.info("Converted controlnet")
    
    if pipe and args.convert_vae_encoder:
        logger.info("Converting vae_encoder")
        convert_vae_encoder(pipe, args)
        logger.info("Converted vae_encoder")
        
    if args.convert_vae_decoder:
        logger.info("Converting vae_decoder")
        convert_vae_decoder(pipe, args)
        logger.info("Converted vae_decoder")

    if args.convert_unet:
        logger.info("Converting unet")
        convert_unet(pipe, args)
        logger.info("Converted unet")

    if args.convert_text_encoder:
        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            logger.info("Converting text_encoder")
            convert_text_encoder(pipe, args)
            logger.info("Converted text_encoder")
        if args.model_is_sdxl:
            logger.info("Converting text_encoder_2")
            convert_text_encoder_2(pipe, args)
            logger.info("Converted text_encoder_2")

    if args.convert_safety_checker:
        logger.info("Converting safety_checker")
        convert_safety_checker(pipe, args)
        logger.info("Converted safety_checker")
        
    if args.quantize_nbits is not None:
        logger.info(f"Quantizing weights to {args.quantize_nbits}-bit precision")
        quantize_weights(args)
        logger.info(f"Quantized weights to {args.quantize_nbits}-bit precision")

    if args.bundle_resources_for_guernika:
        logger.info("Bundling resources for Guernika")
        bundle_resources_for_guernika(pipe, args)
        logger.info("Bundled resources for Guernika")

    if args.clean_up_mlpackages:
        logger.info("Cleaning up MLPackages")
        remove_mlpackages(args)
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
         ))
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
        "The local pre-trained ControlNEt checkpoint and configuration to restore."
    )
    parser.add_argument(
        "--controlnet-checkpoint-location", default=None, type=str, help="Path to the checkpoint to convert."
    )
    parser.add_argument("--compute-unit",
                        choices=tuple(cu
                                      for cu in ct.ComputeUnit._member_names_),
                        default="ALL")

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
        choices=tuple(ai
                      for ai in unet.AttentionImplementations._member_names_),
        default=unet.ATTENTION_IMPLEMENTATION_IN_EFFECT.name,
        help=
        "The enumerated implementations trade off between ANE and GPU performance",
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
