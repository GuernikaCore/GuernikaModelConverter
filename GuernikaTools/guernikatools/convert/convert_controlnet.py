#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools.utils import utils
from guernikatools.models import attention, controlnet

from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)


def main(pipe, args):
    """ Converts the ControlNet component of Stable Diffusion
    """
    if not pipe.controlnet:
        logger.info(f"`controlnet` not available in this pipline.")
        return
    
    out_path = utils.get_out_path(args, "controlnet")
    if os.path.exists(out_path):
        logger.info(f"`controlnet` already exists at {out_path}, skipping conversion.")
        return

    # Register the selected attention implementation globally
    attention.ATTENTION_IMPLEMENTATION_IN_EFFECT = attention.AttentionImplementations[args.attention_implementation]
    logger.info(f"Attention implementation in effect: {attention.ATTENTION_IMPLEMENTATION_IN_EFFECT}")

    # Prepare sample input shapes and values
    batch_size = 2  # for classifier-free guidance
    controlnet_in_channels = pipe.controlnet.config.in_channels
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    height = int(args.output_h / vae_scale_factor)
    width = int(args.output_w / vae_scale_factor)
    
    sample_shape = (
        batch_size,              # B
        controlnet_in_channels,  # C
        height,  # H
        width,  # W
    )
    
    cond_shape = (
        batch_size, # B
        3,          # C
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
        utils.report_correctness(baseline_out, reference_out,  "control baseline to reference PyTorch")

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
        
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_controlnet_inputs, {
            "sample": sample_input_shape,
            "controlnet_cond": cond_input_shape,
        })
    else:
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_controlnet_inputs)
    coreml_controlnet, out_path = utils.convert_to_coreml(
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
    coreml_controlnet.output_description["down_block_res_samples_00"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_01"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_02"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_03"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_04"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_05"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_06"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_07"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["down_block_res_samples_08"] = "Residual down sample from ControlNet"
    if "down_block_res_samples_09" in output_names:
        coreml_controlnet.output_description["down_block_res_samples_09"] = "Residual down sample from ControlNet"
        coreml_controlnet.output_description["down_block_res_samples_10"] = "Residual down sample from ControlNet"
        coreml_controlnet.output_description["down_block_res_samples_11"] = "Residual down sample from ControlNet"
    coreml_controlnet.output_description["mid_block_res_sample"] = "Residual mid sample from ControlNet"
    
    # Set package version metadata
    coreml_controlnet.user_defined_metadata["identifier"] = args.controlnet_version
    coreml_controlnet.user_defined_metadata["converter_version"] = __version__
    coreml_controlnet.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_controlnet.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_controlnet.user_defined_metadata["hidden_size"] = str(hidden_size)
    controlnet_method = utils.conditioning_method_from(args.controlnet_version)
    if controlnet_method:
        coreml_controlnet.user_defined_metadata["method"] = controlnet_method
    
    coreml_controlnet.save(out_path)
    logger.info(f"Saved controlnet into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        coreml_out = list(coreml_controlnet.predict(coreml_sample_controlnet_inputs).values())[0]
        utils.report_correctness(baseline_out, coreml_out, "control baseline PyTorch to reference CoreML")

    del coreml_controlnet
    gc.collect()
