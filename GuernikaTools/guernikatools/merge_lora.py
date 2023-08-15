# Taken from: https://github.com/kohya-ss/sd-scripts/blob/main/networks/merge_lora.py

import math
import argparse
import os
import torch
from safetensors.torch import load_file, save_file
    
# is it possible to apply conv_in and conv_out? -> yes, newer LoCon supports it (^^;)
UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"

# SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
LORA_PREFIX_TEXT_ENCODER_1 = "lora_te1"
LORA_PREFIX_TEXT_ENCODER_2 = "lora_te2"

def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)
    return sd


def save_to_file(file_name, model, state_dict, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name)
    else:
        torch.save(model, file_name)

def merge_to_sd_model(unet, text_encoder, text_encoder_2, models, ratios, merge_dtype=torch.float32):
    unet.to(merge_dtype)
    text_encoder.to(merge_dtype)
    if text_encoder_2:
        text_encoder_2.to(merge_dtype)
    
    layers_per_block = unet.config.layers_per_block

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([unet, text_encoder, text_encoder_2]):
        if not root_module:
            continue
        if i == 0:
            prefix = LORA_PREFIX_UNET
            target_replace_modules = (
                UNET_TARGET_REPLACE_MODULE + UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )
        elif text_encoder_2:
            target_replace_modules = TEXT_ENCODER_TARGET_REPLACE_MODULE
            if i == 1:
                prefix = LORA_PREFIX_TEXT_ENCODER_1
            else:
                prefix = LORA_PREFIX_TEXT_ENCODER_2
        else:
            prefix = LORA_PREFIX_TEXT_ENCODER
            target_replace_modules = TEXT_ENCODER_TARGET_REPLACE_MODULE

        for name, module in root_module.named_modules():
            if module.__class__.__name__ == "LoRACompatibleLinear" or module.__class__.__name__ == "LoRACompatibleConv":
                lora_name = prefix + "." + name
                lora_name = lora_name.replace(".", "_")
                name_to_module[lora_name] = module
            elif module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    for model, ratio in zip(models, ratios):
        print(f"Merging: {model}")
        lora_sd = load_state_dict(model, merge_dtype)

        for key in lora_sd.keys():
            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                # find original module for this lora
                module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                if "input_blocks" in module_name:
                    i = int(module_name.split("input_blocks_", 1)[1].split("_", 1)[0])
                    block_id = (i - 1) // (layers_per_block + 1)
                    layer_in_block_id = (i - 1) % (layers_per_block + 1)
                    module_name = module_name.replace(f"input_blocks_{i}_0", f"down_blocks_{block_id}_resnets_{layer_in_block_id}")
                    module_name = module_name.replace(f"input_blocks_{i}_1", f"down_blocks_{block_id}_attentions_{layer_in_block_id}")
                    module_name = module_name.replace(f"input_blocks_{i}_2", f"down_blocks_{block_id}_resnets_{layer_in_block_id}")
                if "middle_block" in module_name:
                    module_name = module_name.replace("middle_block_0", "mid_block_resnets_0")
                    module_name = module_name.replace("middle_block_1", "mid_block_attentions_0")
                    module_name = module_name.replace("middle_block_2", "mid_block_resnets_1")
                if "output_blocks" in module_name:
                    i = int(module_name.split("output_blocks_", 1)[1].split("_", 1)[0])
                    block_id = i // (layers_per_block + 1)
                    layer_in_block_id = i % (layers_per_block + 1)
                    module_name = module_name.replace(f"output_blocks_{i}_0", f"up_blocks_{block_id}_resnets_{layer_in_block_id}")
                    module_name = module_name.replace(f"output_blocks_{i}_1", f"up_blocks_{block_id}_attentions_{layer_in_block_id}")
                    module_name = module_name.replace(f"output_blocks_{i}_2", f"up_blocks_{block_id}_resnets_{layer_in_block_id}")
                
                module_name = module_name.replace("in_layers_0", "norm1")
                module_name = module_name.replace("in_layers_2", "conv1")

                module_name = module_name.replace("out_layers_0", "norm2")
                module_name = module_name.replace("out_layers_3", "conv2")

                module_name = module_name.replace("emb_layers_1", "time_emb_proj")
                module_name = module_name.replace("skip_connection", "conv_shortcut")

                if module_name not in name_to_module:
                    print(f"no module found for LoRA weight: {key}")
                    continue
                module = name_to_module[module_name]
                # print(f"apply {key} to {module}")

                down_weight = lora_sd[key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                # W <- W + U * D
                weight = module.weight
                # print(module_name, down_weight.size(), up_weight.size())
                if len(weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                module.weight = torch.nn.Parameter(weight)


def merge_lora_models(models, ratios, merge_dtype):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    for model, ratio in zip(models, ratios):
        print(f"loading: {model}")
        lora_sd = load_state_dict(model, merge_dtype)

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        print(f"merging...")
        for key in lora_sd.keys():
            if "alpha" in key:
                continue

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size()
                ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd
