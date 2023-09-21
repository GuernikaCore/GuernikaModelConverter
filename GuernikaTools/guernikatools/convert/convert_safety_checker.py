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

from types import MethodType


def main(pipe, args):
    """ Converts the Safety Checker component of Stable Diffusion
    """
    if pipe.safety_checker is None:
        logger.warning(
            f"diffusers pipeline for {args.model_version} does not have a `safety_checker` module! " \
            "`--convert-safety-checker` will be ignored."
        )
        return

    out_path = utils.get_out_path(args, "safety_checker")
    if os.path.exists(out_path):
        logger.info(f"`safety_checker` already exists at {out_path}, skipping conversion.")
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
        
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_safety_checker_inputs, {
            "clip_input": clip_input_shape, "images": input_shape
        })
    else:
        sample_coreml_inputs = utils.get_coreml_inputs(coreml_sample_safety_checker_inputs)
    coreml_safety_checker, out_path = utils.convert_to_coreml(
        "safety_checker", traced_safety_checker,
        sample_coreml_inputs,
        ["filtered_images", "has_nsfw_concepts", "concept_scores"], False, args
    )

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
        utils.report_correctness(baseline_out, coreml_out, "safety_checker baseline PyTorch to reference CoreML")

    del traced_safety_checker, coreml_safety_checker, pipe.safety_checker
    gc.collect()
