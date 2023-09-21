#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
        
from guernikatools._version import __version__
from guernikatools.utils import utils
from guernikatools.models import attention

from diffusers import T2IAdapter

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


def main(base_adapter, args):
    """ Converts a T2IAdapter
    """
    out_path = utils.get_out_path(args, "t2i_adapter")
    if os.path.exists(out_path):
        logger.info(f"`t2i_adapter` already exists at {out_path}, skipping conversion.")
        return

    # Register the selected attention implementation globally
    attention.ATTENTION_IMPLEMENTATION_IN_EFFECT = attention.AttentionImplementations[args.attention_implementation]
    logger.info(f"Attention implementation in effect: {attention.ATTENTION_IMPLEMENTATION_IN_EFFECT}")

    # Prepare sample input shapes and values
    batch_size = 2  # for classifier-free guidance
    adapter_type = base_adapter.config.adapter_type
    adapter_in_channels = base_adapter.config.in_channels
    
    input_shape = (
        1,                    # B
        adapter_in_channels,  # C
        args.output_h,  # H
        args.output_w,  # W
    )
    
    sample_adapter_inputs = {
        "input": torch.rand(*input_shape, dtype=torch.float16)
    }
    sample_adapter_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_adapter_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_adapter_inputs_spec}")

    # Initialize reference adapter
    reference_adapter = T2IAdapter(**base_adapter.config).eval()
    load_state_dict_summary = reference_adapter.load_state_dict(base_adapter.state_dict())

    # Prepare inputs
    baseline_sample_adapter_inputs = deepcopy(sample_adapter_inputs)

    # JIT trace
    logger.info("JIT tracing..")
    reference_adapter = torch.jit.trace(reference_adapter, (sample_adapter_inputs["input"].to(torch.float32), ))
    logger.info("Done.")

    if args.check_output_correctness:
        baseline_out = base_adapter(**baseline_sample_adapter_inputs, return_dict=False)[0].numpy()
        reference_out = reference_adapter(**sample_adapter_inputs)[0].numpy()
        utils.report_correctness(baseline_out, reference_out,  "control baseline to reference PyTorch")

    del base_adapter
    gc.collect()

    coreml_sample_adapter_inputs = {
        k: v.numpy().astype(np.float16)
        for k, v in sample_adapter_inputs.items()
    }
    
    if args.multisize:
        input_size = args.output_h
        input_shape = ct.Shape(shape=(
            1,
            adapter_in_channels,
            ct.RangeDim(int(input_size * 0.5), upper_bound=int(input_size * 2), default=input_size),
            ct.RangeDim(int(input_size * 0.5), upper_bound=int(input_size * 2), default=input_size)
        ))
        
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_adapter_inputs, {"input": input_shape})
    else:
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_adapter_inputs)
    output_names = [
        "adapter_res_samples_00", "adapter_res_samples_01",
        "adapter_res_samples_02", "adapter_res_samples_03"
    ]
    coreml_adapter, out_path = utils.convert_to_coreml(
        "t2i_adapter",
        reference_adapter,
        sample_coreml_inputs,
        output_names,
        args.precision_full,
        args
    )
    del reference_adapter
    gc.collect()

    # Set model metadata
    coreml_adapter.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_adapter.license = "T2IAdapter (https://github.com/TencentARC/T2I-Adapter)"
    coreml_adapter.version = args.t2i_adapter_version
    coreml_adapter.short_description = \
        "T2IAdapter is a neural network structure to control diffusion models by adding extra conditions. " \
        "Please refer to https://github.com/TencentARC/T2I-Adapter for details."

    # Set the input descriptions
    coreml_adapter.input_description["input"] = "Image used to condition adapter output"

    # Set the output descriptions
    coreml_adapter.output_description["adapter_res_samples_00"] = "Residual sample from T2IAdapter"
    coreml_adapter.output_description["adapter_res_samples_01"] = "Residual sample from T2IAdapter"
    coreml_adapter.output_description["adapter_res_samples_02"] = "Residual sample from T2IAdapter"
    coreml_adapter.output_description["adapter_res_samples_03"] = "Residual sample from T2IAdapter"
    
    # Set package version metadata
    coreml_adapter.user_defined_metadata["identifier"] = args.t2i_adapter_version
    coreml_adapter.user_defined_metadata["converter_version"] = __version__
    coreml_adapter.user_defined_metadata["attention_implementation"] = args.attention_implementation
    coreml_adapter.user_defined_metadata["compute_unit"] = args.compute_unit
    coreml_adapter.user_defined_metadata["adapter_type"] = adapter_type
    adapter_method = conditioning_method_from(args.t2i_adapter_version)
    if adapter_method:
        coreml_adapter.user_defined_metadata["method"] = adapter_method
    
    coreml_adapter.save(out_path)
    logger.info(f"Saved adapter into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        coreml_out = list(coreml_adapter.predict(coreml_sample_adapter_inputs).values())[0]
        utils.report_correctness(baseline_out, coreml_out, "control baseline PyTorch to reference CoreML")

    del coreml_adapter
    gc.collect()
