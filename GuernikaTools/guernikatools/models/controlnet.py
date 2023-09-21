#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from guernikatools.models import attention
from .layer_norm import LayerNormANE
from .unet import Timesteps, TimestepEmbedding, UNetMidBlock2DCrossAttn
from .unet import linear_to_conv2d_map, get_down_block, get_up_block

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin

from enum import Enum

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure minimum macOS version requirement is met for this particular model
from coremltools.models.utils import _macos_version
if not _macos_version() >= (13, 1):
    logger.warning(
        "!!! macOS 13.1 and newer or iOS/iPadOS 16.2 and newer is required for best performance !!!"
    )

class ControlNetConditioningDefaultEmbedding(nn.Module):
    """
    "Stable Diffusion uses a pre-processing method similar to VQ-GAN [11] to convert the entire dataset of 512 × 512
    images into smaller 64 × 64 “latent images” for stabilized training. This requires ControlNets to convert
    image-based conditions to 64 × 64 feature space to match the convolution size. We use a tiny network E(·) of four
    convolution layers with 4 × 4 kernels and 2 × 2 strides (activated by ReLU, channels are 16, 32, 64, 128,
    initialized with Gaussian weights, trained jointly with the full model) to encode image-space conditions ... into
    feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels = (16, 32, 96, 256),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        return embedding

class ControlNetModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        sample_size=None,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        only_cross_attention=False,
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=768,
        attention_head_dim=8,
        transformer_layers_per_block=1,
        conv_in_kernel=3,
        conv_out_kernel=3,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        projection_class_embeddings_input_dim=None,
        conditioning_embedding_out_channels=(16, 32, 96, 256),
        **kwargs,
    ):
        if kwargs.get("dual_cross_attention", None):
            raise NotImplementedError
        if kwargs.get("num_classs_embeds", None):
            raise NotImplementedError
        if only_cross_attention:
            raise NotImplementedError
        if kwargs.get("use_linear_projection", None):
            logger.warning("`use_linear_projection=True` is ignored!")

        super().__init__()
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding
        )

        # time
        time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        
        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type is not None:
            raise NotImplementedError

        self.time_proj = time_proj
        self.time_embedding = time_embedding
        
        self.controlnet_cond_embedding = ControlNetConditioningDefaultEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)
        
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[i],
            resnet_groups=norm_num_groups,
        )

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        text_embeds=None,
        time_ids=None
    ):
        # 0. Project (or look-up) time embeddings
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)
        
        if hasattr(self.config, "addition_embed_type") and self.config.addition_embed_type == "text_time":
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb

        # 1. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        sample += controlnet_cond

        # 3. down
        down_block_res_samples = (sample, )
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
        
        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample, )

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        return (down_block_res_samples, mid_block_res_sample)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
