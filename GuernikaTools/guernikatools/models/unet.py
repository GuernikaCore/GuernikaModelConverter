#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from .attention import Attention
from .layer_norm import LayerNormANE

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        if 'weight' in k and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]

# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix + "bias"] = state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
    return state_dict


class LayerNormANE(LayerNormANE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)


# Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py
# (modified, e.g. the attention implementation)
class CrossAttnUpBlock2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        attn_num_head_channels=1,
        cross_attention_dim=768,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=transformer_layers_per_block,
                    context_dim=cross_attention_dim,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])

    def forward(self,
                hidden_states,
                res_hidden_states_tuple,
                temb=None,
                encoder_hidden_states=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):

    def __init__(
        self,
        in_channels,
        prev_output_channel,
        out_channels,
        temb_channels,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CrossAttnDownBlock2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        attn_num_head_channels=1,
        cross_attention_dim=768,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=transformer_layers_per_block,
                    context_dim=cross_attention_dim,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, additional_residuals=None):
        output_states = ()
        
        blocks = list(zip(self.resnets, self.attentions))
        for i, (resnet, attn) in enumerate(blocks):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals
            
            output_states += (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states, )

        return hidden_states, output_states


class DownBlock2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states, )

        return hidden_states, output_states


class ResnetBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        temb_channels=512,
        groups=32,
        groups_out=None,
        eps=1e-6,
        time_embedding_norm="default",
        use_nin_shortcut=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True
        )

        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Conv2d(
                temb_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.nonlinearity = nn.SiLU()

        self.use_nin_shortcut = self.in_channels != self.out_channels if use_nin_shortcut is None else use_nin_shortcut

        self.conv_shortcut = None
        if self.use_nin_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x, temb):
        hidden_states = x
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = (x + hidden_states)

        return out


class Upsample2D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample2D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class SpatialTransformer(nn.Module):

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        context_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True
        )

        self.proj_in = nn.Conv2d(
            in_channels,
            inner_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                context_dim=context_dim
            )
            for d in range(depth)
        ])

        self.proj_out = nn.Conv2d(
            inner_dim,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, hidden_states, context=None):
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.view(batch, channel, 1, height * weight)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)
        hidden_states = hidden_states.view(batch, channel, height, weight)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
        )
        self.ff = FeedForward(dim, glu=gated_ff)
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
        )
        self.norm1 = LayerNormANE(dim)
        self.norm2 = LayerNormANE(dim)
        self.norm3 = LayerNormANE(dim)

    def forward(self, hidden_states, context=None):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim_in=dim, dim_out=inner_dim), nn.Identity(),
            nn.Conv2d(
                inner_dim,
                dim_out if dim_out is not None else dim,
                kernel_size=1)
            )

    def forward(self, hidden_states):
        return self.net(hidden_states)


class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out * 2, kernel_size=1)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=1)
        return hidden_states * F.gelu(gate)


class TimestepEmbedding(nn.Module):

    def __init__(self, channel, time_embed_dim, act_fn="silu"):
        super().__init__()

        self.linear_1 = nn.Conv2d(channel, time_embed_dim, kernel_size=1)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Conv2d(
            time_embed_dim,
            time_embed_dim,
            kernel_size=1
        )

    def forward(self, sample):
        if len(sample.shape) == 2:
            sample = sample.unsqueeze(-1).unsqueeze(-1)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):

    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


def get_timestep_embedding(
    timesteps,
    embedding_dim,
    flip_sin_to_cos=False,
    downscale_freq_shift=1,
    scale=1,
    max_period=10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class UNetMidBlock2DCrossAttn(nn.Module):

    def __init__(
        self,
        in_channels,
        temb_channels,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        attn_num_head_channels=1,
        attention_type="default",
        cross_attention_dim=768,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                time_embedding_norm=resnet_time_scale_shift,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                SpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=transformer_layers_per_block,
                    context_dim=cross_attention_dim,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNet2DConditionModel(ModelMixin, ConfigMixin):

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
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
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
        **kwargs,
    ):
        if kwargs.get("dual_cross_attention", None):
            raise NotImplementedError
        if kwargs.get("num_class_embeds", None):
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

        self.time_proj = time_proj
        self.time_embedding = time_embedding
        
        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type is not None:
            raise NotImplementedError
        
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
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

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[i],
            resnet_groups=norm_num_groups,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding
        )

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds=None,
        time_ids=None,
        adapter_res_samples_00=None,
        adapter_res_samples_01=None,
        adapter_res_samples_02=None,
        adapter_res_samples_03=None,
        down_block_res_samples_00=None,
        down_block_res_samples_01=None,
        down_block_res_samples_02=None,
        down_block_res_samples_03=None,
        down_block_res_samples_04=None,
        down_block_res_samples_05=None,
        down_block_res_samples_06=None,
        down_block_res_samples_07=None,
        down_block_res_samples_08=None,
        down_block_res_samples_09=None,
        down_block_res_samples_10=None,
        down_block_res_samples_11=None,
        mid_block_res_sample=None,
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

        # 3. down
        is_adapter = adapter_res_samples_00 is not None
        is_controlnet = mid_block_res_sample is not None and down_block_res_samples_00 is not None
        adapter_residuals = [
            adapter_res_samples_00,
            adapter_res_samples_01,
            adapter_res_samples_02,
            adapter_res_samples_03
        ]
        
        down_block_res_samples = (sample, )
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = None
                if is_adapter and len(adapter_residuals) > 0:
                    additional_residuals = adapter_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    additional_residuals=additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(adapter_residuals) > 0:
                    sample += adapter_residuals.pop(0)

            down_block_res_samples += res_samples
        
        if is_controlnet:
            down_block_additional_residuals = (
                down_block_res_samples_00,
                down_block_res_samples_01,
                down_block_res_samples_02,
                down_block_res_samples_03,
                down_block_res_samples_04,
                down_block_res_samples_05,
                down_block_res_samples_06,
                down_block_res_samples_07,
                down_block_res_samples_08,
                down_block_res_samples_09,
                down_block_res_samples_10,
                down_block_res_samples_11,
            )
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
        
        # 4. mid
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states
        )
        
        if is_controlnet:
            sample += mid_block_res_sample

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block,"attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return (sample, )


def get_down_block(
    down_block_type,
    num_layers,
    transformer_layers_per_block,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    cross_attention_dim=None,
    downsample_padding=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock2D"
            )
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
        )


def get_up_block(
    up_block_type,
    num_layers,
    transformer_layers_per_block,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    cross_attention_dim=None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
        )
    raise ValueError(f"{up_block_type} does not exist.")
