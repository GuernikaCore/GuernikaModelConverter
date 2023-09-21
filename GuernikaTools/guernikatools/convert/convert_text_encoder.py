#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools.utils import utils

from collections import OrderedDict, defaultdict
from copy import deepcopy
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


def main(tokenizer, text_encoder, args, model_name="text_encoder"):
    """ Converts the text encoder component of Stable Diffusion
    """
    out_path = utils.get_out_path(args, model_name)
    if os.path.exists(out_path):
        logger.info(
            f"`text_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = tokenizer.model_max_length
    text_encoder_hidden_size = text_encoder.config.hidden_size

    sample_text_encoder_inputs = {
        "input_ids":
        torch.randint(
            text_encoder.config.vocab_size,
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
            self.text_encoder = text_encoder

        def forward(self, input_ids):
            return text_encoder(input_ids, return_dict=False)

    class TextEncoderXL(nn.Module):

        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, input_ids):
            output = text_encoder(input_ids, output_hidden_states=True)
            return (output.hidden_states[-2], output[0])
    
    reference_text_encoder = TextEncoderXL().eval() if args.model_is_sdxl else TextEncoder().eval()

    logger.info("JIT tracing {model_name}..")
    reference_text_encoder = torch.jit.trace(
        reference_text_encoder,
        (sample_text_encoder_inputs["input_ids"].to(torch.int32), ),
    )
    logger.info("Done.")

    sample_coreml_inputs = utils.get_coreml_inputs(sample_text_encoder_inputs)
    coreml_text_encoder, out_path = utils.convert_to_coreml(
        model_name, reference_text_encoder, sample_coreml_inputs,
        ["last_hidden_state", "pooled_outputs"], args.precision_full, args
    )

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
    coreml_text_encoder.user_defined_metadata["hidden_size"] = str(text_encoder.config.hidden_size)

    coreml_text_encoder.save(out_path)

    logger.info(f"Saved {model_name} into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = text_encoder(
            sample_text_encoder_inputs["input_ids"].to(torch.int32),
            return_dict=False,
        )[1].numpy()

        coreml_out = list(coreml_text_encoder.predict({
            k: v.numpy() for k, v in sample_text_encoder_inputs.items()
        }).values())[0]
        utils.report_correctness(baseline_out, coreml_out, "{model_name} baseline PyTorch to reference CoreML")

    del reference_text_encoder, coreml_text_encoder, text_encoder
    gc.collect()
