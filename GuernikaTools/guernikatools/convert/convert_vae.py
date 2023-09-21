#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools.utils import utils

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


def encoder(pipe, args):
    """ Converts the VAE Encoder component of Stable Diffusion
    """
    out_path = utils.get_out_path(args, "vae_encoder")
    if os.path.exists(out_path):
        logger.info(f"`vae_encoder` already exists at {out_path}, skipping conversion.")
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
    
    # TODO: For now using variable size takes too much memory and time
#    if args.multisize:
#        sample_size = args.output_h
#        input_shape = ct.Shape(shape=(
#            1,
#            3,
#            ct.RangeDim(lower_bound=int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
#            ct.RangeDim(lower_bound=int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size)
#        ))
#        sample_coreml_inputs = utils.get_coreml_inputs(sample_vae_encoder_inputs, {"z": input_shape})
#    else:
#        sample_coreml_inputs = utils.get_coreml_inputs(sample_vae_encoder_inputs)
    sample_coreml_inputs = utils.get_coreml_inputs(sample_vae_encoder_inputs)
    
    # SDXL seems to require full precision
    precision_full = args.model_is_sdxl and args.model_version != "stabilityai/stable-diffusion-xl-base-1.0"
    coreml_vae_encoder, out_path = utils.convert_to_coreml(
        "vae_encoder", traced_vae_encoder, sample_coreml_inputs,
        ["latent"], precision_full, args
    )

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
    # TODO: Add this key to stop using the hack when variable size works correctly
    coreml_vae_encoder.user_defined_metadata["supports_model_size_hack"] = "true"

    coreml_vae_encoder.save(out_path)

    logger.info(f"Saved vae_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_encoder(z=sample_vae_encoder_inputs["z"].to(torch.float32)).detach().numpy()
        coreml_out = list(coreml_vae_encoder.predict({
            k: v.numpy() for k, v in sample_vae_encoder_inputs.items()
        }).values())[0]
        utils.report_correctness(baseline_out, coreml_out,"vae_encoder baseline PyTorch to baseline CoreML")

    del traced_vae_encoder, pipe.vae.encoder, coreml_vae_encoder
    gc.collect()


def decoder(pipe, args):
    """ Converts the VAE Decoder component of Stable Diffusion
    """
    out_path = utils.get_out_path(args, "vae_decoder")
    if os.path.exists(out_path):
        logger.info(f"`vae_decoder` already exists at {out_path}, skipping conversion.")
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
    
    # TODO: For now using variable size takes too much memory and time
#    if args.multisize:
#        sample_size = height
#        input_shape = ct.Shape(shape=(
#            1,
#            vae_latent_channels,
#            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size),
#            ct.RangeDim(int(sample_size * 0.5), upper_bound=int(sample_size * 2), default=sample_size)
#        ))
#        sample_coreml_inputs = utils.get_coreml_inputs(sample_vae_decoder_inputs, {"z": input_shape})
#    else:
#        sample_coreml_inputs = utils.get_coreml_inputs(sample_vae_decoder_inputs)
    sample_coreml_inputs = utils.get_coreml_inputs(sample_vae_decoder_inputs)
    
    # SDXL seems to require full precision
    precision_full = args.model_is_sdxl and args.model_version != "stabilityai/stable-diffusion-xl-base-1.0"
    coreml_vae_decoder, out_path = utils.convert_to_coreml(
        "vae_decoder", traced_vae_decoder, sample_coreml_inputs,
        ["image"], precision_full, args
    )

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
    # TODO: Add this key to stop using the hack when variable size works correctly
    coreml_vae_decoder.user_defined_metadata["supports_model_size_hack"] = "true"

    coreml_vae_decoder.save(out_path)

    logger.info(f"Saved vae_decoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_decoder(z=sample_vae_decoder_inputs["z"].to(torch.float32)).detach().numpy()
        coreml_out = list(coreml_vae_decoder.predict({
            k: v.numpy() for k, v in sample_vae_decoder_inputs.items()
        }).values())[0]
        utils.report_correctness(baseline_out, coreml_out, "vae_decoder baseline PyTorch to baseline CoreML")

    del traced_vae_decoder, pipe.vae.decoder, coreml_vae_decoder
    gc.collect()
