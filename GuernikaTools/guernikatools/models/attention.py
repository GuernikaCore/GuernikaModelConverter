import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import torch.nn as nn

from enum import Enum


# Ensure minimum macOS version requirement is met for this particular model
from coremltools.models.utils import _macos_version
if not _macos_version() >= (13, 1):
    logger.warning(
        "!!! macOS 13.1 and newer or iOS/iPadOS 16.2 and newer is required for best performance !!!"
    )


class AttentionImplementations(Enum):
    ORIGINAL = "ORIGINAL"
    SPLIT_EINSUM = "SPLIT_EINSUM"
    SPLIT_EINSUM_V2 = "SPLIT_EINSUM_V2"


ATTENTION_IMPLEMENTATION_IN_EFFECT = AttentionImplementations.SPLIT_EINSUM_V2


WARN_MSG = \
    "This `nn.Module` is intended for Apple Silicon deployment only. " \
    "PyTorch-specific optimizations and training is disabled"

class Attention(nn.Module):
    """ Apple Silicon friendly version of `diffusers.models.attention.Attention`
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=1, bias=bias)
        self.to_k = nn.Conv2d(context_dim, inner_dim, kernel_size=1, bias=bias)
        self.to_v = nn.Conv2d(context_dim, inner_dim, kernel_size=1, bias=bias)
        if dropout > 0:
            self.to_out = nn.Sequential(
                nn.Conv2d(inner_dim, query_dim, kernel_size=1, bias=True), nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Sequential(
                nn.Conv2d(inner_dim, query_dim, kernel_size=1, bias=True)
            )


    def forward(self, hidden_states, encoder_hidden_states=None, mask=None):
        if self.training:
            raise NotImplementedError(WARN_MSG)

        batch_size, dim, _, sequence_length = hidden_states.shape

        q = self.to_q(hidden_states)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Validate mask
        if mask is not None:
            expected_mask_shape = [batch_size, sequence_length, 1, 1]
            if mask.dtype == torch.bool:
                mask = mask.logical_not().float() * -1e4
            elif mask.dtype == torch.int64:
                mask = (1 - mask).float() * -1e4
            elif mask.dtype != torch.float32:
                raise TypeError(f"Unexpected dtype for mask: {mask.dtype}")

            if len(mask.size()) == 2:
                mask = mask.unsqueeze(2).unsqueeze(2)

            if list(mask.size()) != expected_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `mask` (Expected {expected_mask_shape}, got {list(mask.size())}"
                )

        if ATTENTION_IMPLEMENTATION_IN_EFFECT == AttentionImplementations.ORIGINAL:
            attn = original(q, k, v, mask, self.heads, self.dim_head)
        elif ATTENTION_IMPLEMENTATION_IN_EFFECT == AttentionImplementations.SPLIT_EINSUM:
            attn = split_einsum(q, k, v, mask, self.heads, self.dim_head)
        elif ATTENTION_IMPLEMENTATION_IN_EFFECT == AttentionImplementations.SPLIT_EINSUM_V2:
            attn = split_einsum_v2(q, k, v, mask, self.heads, self.dim_head)
        else:
            raise ValueError(ATTENTION_IMPLEMENTATION_IN_EFFECT)

        return self.to_out(attn)

def split_einsum(q, k, v, mask, heads, dim_head):
    """ Attention Implementation backing AttentionImplementations.SPLIT_EINSUM

    - Implements https://machinelearning.apple.com/research/neural-engine-transformers
    - Recommended for ANE
    - Marginally slower on GPU
    """
    mh_q = [
        q[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    k = k.transpose(1, 3)
    mh_k = [
        k[:, :, :,
          head_idx * dim_head:(head_idx + 1) * dim_head]
        for head_idx in range(heads)
    ]  # (bs, max_seq_length, 1, dim_head) * heads

    mh_v = [
        v[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    attn_weights = [
        torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * (dim_head**-0.5)
        for qi, ki in zip(mh_q, mh_k)
    ]  # (bs, max_seq_length, 1, max_seq_length) * heads

    if mask is not None:
        for head_idx in range(heads):
            attn_weights[head_idx] = attn_weights[head_idx] + mask

    attn_weights = [
        aw.softmax(dim=1) for aw in attn_weights
    ]  # (bs, max_seq_length, 1, max_seq_length) * heads
    attn = [
        torch.einsum("bkhq,bchk->bchq", wi, vi)
        for wi, vi in zip(attn_weights, mh_v)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    attn = torch.cat(attn, dim=1)  # (bs, dim, 1, max_seq_length)
    return attn


CHUNK_SIZE = 512

def split_einsum_v2(q, k, v, mask, heads, dim_head):
    """ Attention Implementation backing AttentionImplementations.SPLIT_EINSUM_V2

    - Implements https://machinelearning.apple.com/research/neural-engine-transformers
    - Recommended for ANE
    - Marginally slower on GPU
    - Chunks the query sequence to avoid large intermediate tensors and improves ANE performance
    """
    query_seq_length = q.size(3)
    num_chunks = query_seq_length // CHUNK_SIZE
    
    if num_chunks == 0:
        logger.info(
            "AttentionImplementations.SPLIT_EINSUM_V2: query sequence too short to chunk "
            f"({query_seq_length}<{CHUNK_SIZE}), fall back to AttentionImplementations.SPLIT_EINSUM (safe to ignore)")
        return split_einsum(q, k, v, mask, heads, dim_head)
    
    logger.info(
        "AttentionImplementations.SPLIT_EINSUM_V2: Splitting query sequence length of "
        f"{query_seq_length} into {num_chunks} chunks")

    mh_q = [
        q[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    # Chunk the query sequence for each head
    mh_q_chunked = [
        [h_q[..., chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE] for chunk_idx in range(num_chunks)]
        for h_q in mh_q
    ]  # ((bs, dim_head, 1, QUERY_SEQ_CHUNK_SIZE) * num_chunks) * heads

    k = k.transpose(1, 3)
    mh_k = [
        k[:, :, :,
          head_idx * dim_head:(head_idx + 1) * dim_head]
        for head_idx in range(heads)
    ]  # (bs, max_seq_length, 1, dim_head) * heads

    mh_v = [
        v[:, head_idx * dim_head:(head_idx + 1) *
          dim_head, :, :] for head_idx in range(heads)
    ]  # (bs, dim_head, 1, max_seq_length) * heads

    attn_weights = [
        [
            torch.einsum("bchq,bkhc->bkhq", [qi_chunk, ki]) * (dim_head**-0.5)
            for qi_chunk in h_q_chunked
        ] for h_q_chunked, ki in zip(mh_q_chunked, mh_k)
    ]  # ((bs, max_seq_length, 1, chunk_size) * num_chunks) * heads

    attn_weights = [
        [aw_chunk.softmax(dim=1) for aw_chunk in aw_chunked]
        for aw_chunked in attn_weights
    ]  # ((bs, max_seq_length, 1, chunk_size) * num_chunks) * heads

    attn = [
        [
            torch.einsum("bkhq,bchk->bchq", wi_chunk, vi)
            for wi_chunk in wi_chunked
        ] for wi_chunked, vi in zip(attn_weights, mh_v)
    ]  # ((bs, dim_head, 1, chunk_size) * num_chunks) * heads

    attn = torch.cat([
        torch.cat(attn_chunked, dim=3) for attn_chunked in attn
    ], dim=1)  # (bs, dim, 1, max_seq_length)

    return attn


def original(q, k, v, mask, heads, dim_head):
    """ Attention Implementation backing AttentionImplementations.ORIGINAL

    - Not recommended for ANE
    - Recommended for GPU
    """
    bs = q.size(0)
    mh_q = q.view(bs, heads, dim_head, -1)
    mh_k = k.view(bs, heads, dim_head, -1)
    mh_v = v.view(bs, heads, dim_head, -1)

    attn_weights = torch.einsum("bhcq,bhck->bhqk", [mh_q, mh_k])
    attn_weights.mul_(dim_head**-0.5)

    if mask is not None:
        attn_weights = attn_weights + mask

    attn_weights = attn_weights.softmax(dim=3)

    attn = torch.einsum("bhqk,bhck->bhcq", [attn_weights, mh_v])
    attn = attn.contiguous().view(bs, heads * dim_head, 1, -1)
    return attn
