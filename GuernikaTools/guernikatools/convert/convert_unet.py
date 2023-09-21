#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools.utils import utils, chunk_mlprogram
from guernikatools.models import attention, unet

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
    """ Converts the UNet component of Stable Diffusion
    """
    out_path = utils.get_out_path(args, "unet")

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
        attention.ATTENTION_IMPLEMENTATION_IN_EFFECT = attention.AttentionImplementations[args.attention_implementation]
        logger.info(f"Attention implementation in effect: {attention.ATTENTION_IMPLEMENTATION_IN_EFFECT}")

        # Prepare sample input shapes and values
        batch_size = 2  # for classifier-free guidance
        unet_in_channels = pipe.unet.config.in_channels
        # allow converting instruct pix2pix
        if unet_in_channels == 8:
            batch_size = 3
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = int(args.output_h / vae_scale_factor)
        width = int(args.output_w / vae_scale_factor)
        
        sample_shape = (
            batch_size,        # B
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

        args.hidden_size = pipe.unet.config.cross_attention_dim
        encoder_hidden_states_shape = (
            batch_size,
            args.hidden_size,
            1,
            max_position_embeddings,
        )

        # Create the scheduled timesteps for downstream use
        DEFAULT_NUM_INFERENCE_STEPS = 50
        pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)
        
        sample_unet_inputs = [
            ("sample", torch.rand(*sample_shape)),
            ("timestep", torch.tensor([pipe.scheduler.timesteps[0].item()] * (batch_size)).to(torch.float32)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ]
        
        args.requires_aesthetics_score = False
        if hasattr(pipe.unet.config, "addition_embed_type") and pipe.unet.config.addition_embed_type == "text_time":
            text_embeds_shape = (
                batch_size,
                pipe.text_encoder_2.config.hidden_size,
            )
            time_ids_input = None
            if hasattr(pipe.config, "requires_aesthetics_score") and pipe.config.requires_aesthetics_score:
                args.requires_aesthetics_score = True
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
        
        if args.t2i_adapter_support:
            block_out_channels = pipe.unet.config.block_out_channels
            
            if args.model_is_sdxl:
                t2ia_height = int(height / 2)
                t2ia_width = int(width / 2)
            else:
                t2ia_height = height
                t2ia_width = width
            
            for i, output_channel in enumerate(block_out_channels):
                sample_unet_inputs = sample_unet_inputs + [
                    (f"adapter_res_samples_{i:02}", torch.rand(2,  output_channel, t2ia_height, t2ia_width)),
                ]
                if not args.model_is_sdxl or i == 1:
                    t2ia_height = int(t2ia_height / 2)
                    t2ia_width = int(t2ia_width / 2)
        
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
                if "res_sample" in k:
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
            baseline_out = pipe.unet(
                sample=baseline_sample_unet_inputs["sample"],
                timestep=baseline_sample_unet_inputs["timestep"],
                encoder_hidden_states=baseline_sample_unet_inputs["encoder_hidden_states"],
                return_dict=False
            )[0].detach().numpy()
            reference_out = reference_unet(**sample_unet_inputs)[0].detach().numpy()
            utils.report_correctness(baseline_out, reference_out, "unet baseline to reference PyTorch")

            del pipe.unet
            gc.collect()

        coreml_sample_unet_inputs = {
            k: v.numpy().astype(np.float16)
            for k, v in sample_unet_inputs.items()
        }
        
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_unet_inputs, multisize_inputs)
        precision_full = args.precision_full
        if not precision_full and pipe.scheduler.config.prediction_type == "v_prediction":
            precision_full = True
            logger.info(f"Full precision required: prediction_type == v_prediction")
        coreml_unet, out_path = utils.convert_to_coreml(
            "unet",
            reference_unet,
            sample_coreml_inputs,
            ["noise_pred"],
            precision_full,
            args
        )
        del reference_unet
        gc.collect()

        update_coreml_unet(pipe, coreml_unet, out_path, args)
        logger.info(f"Saved unet into {out_path}")

        # Parity check PyTorch vs CoreML
        if args.check_output_correctness:
            coreml_out = list(coreml_unet.predict(coreml_sample_unet_inputs).values())[0]
            utils.report_correctness(baseline_out, coreml_out, "unet baseline PyTorch to reference CoreML")

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


def update_coreml_unet(pipe, coreml_unet, out_path, args):
    # make ControlNet/T2IAdapter inputs optional
    coreml_spec = coreml_unet.get_spec()
    for index, input_spec in enumerate(coreml_spec.description.input):
        if "res_sample" in input_spec.name:
            coreml_spec.description.input[index].type.isOptional = True
    coreml_unet = ct.models.MLModel(coreml_spec, skip_model_load=True, weights_dir=coreml_unet.weights_dir)
    
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
        coreml_unet.input_description["text_embeds"] = ""
        coreml_unet.input_description["time_ids"] = ""
    if args.t2i_adapter_support:
        coreml_unet.input_description["adapter_res_samples_00"] = "Optional: Residual down sample from T2IAdapter"
    if args.controlnet_support:
        coreml_unet.input_description["down_block_res_samples_00"] = "Optional: Residual down sample from ControlNet"
        coreml_unet.input_description["mid_block_res_sample"] = "Optional: Residual mid sample from ControlNet"

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
    coreml_unet.user_defined_metadata["hidden_size"] = str(args.hidden_size)
    if args.requires_aesthetics_score:
        coreml_unet.user_defined_metadata["requires_aesthetics_score"] = "true"
    
    coreml_unet.save(out_path)
