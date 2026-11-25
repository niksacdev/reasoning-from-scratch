"""Padding-aware generation using LFM3 attention and convolution caches."""

import torch

from .lfm3 import KVCache, LFM3Model as LFM3Model, LFM3Tokenizer as LFM3Tokenizer, LFM3_CONFIG_1_7B as LFM3_CONFIG_1_7B
from .lfm3 import download_lfm3 as download_lfm3, load_checkpoint as load_checkpoint
from .lfm3 import load_model_and_tokenizer as _load_model_and_tokenizer

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope_with_pos_ids(x, cos, sin, pos_ids):
    B, H, L, D = x.shape
    cos_sel = cos[pos_ids]  # (B, L, D)
    sin_sel = sin[pos_ids]  # (B, L, D)
    cos_sel = cos_sel.unsqueeze(1)  # (B, 1, L, D)
    sin_sel = sin_sel.unsqueeze(1)  # (B, 1, L, D)
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_sel) + (rotated * sin_sel)
    return x_rotated.to(dtype=x.dtype)


@torch.inference_mode()
def generate_text_basic_batched_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    device = token_ids.device
    model.eval()

    batch_size, input_length = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # Init cache and model position
    cache = KVCache(n_layers=model.cfg["n_layers"])

    # Prefill
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]

    # Track which sequences have already produced EOS
    if eos_token_id is not None:
        # If a prompt already ends with EOS, consider it finished
        finished = (token_ids[:, -1] == eos_token_id)
    else:
        finished = None

    # Decode
    cur_attn = attn_mask
    generated_tokens = []
    for step in range(max_new_tokens):
        # If all sequences are already finished, stop
        if eos_token_id is not None and finished is not None and torch.all(finished):
            break

        if step > 0:
            # Extend mask to include the newly generated token
            if cur_attn is not None:
                ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
                cur_attn = torch.cat([cur_attn, ones], dim=1)

            # Advance one token with KV cache
            out = model(generated_tokens[-1], cache=cache, attn_mask=cur_attn)[:, -1]

        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None:
            # Force already finished rows to keep emitting EOS to maintain shape
            eos_tok = next_token.new_full((batch_size, 1), eos_token_id)
            next_token = torch.where(finished.view(batch_size, 1), eos_tok, next_token)

        generated_tokens.append(next_token)

        # Update finished mask after appending this step's token
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_token_id)

    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    return token_ids[:, input_length:]


@torch.inference_mode()
def generate_text_basic_batched_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    device = token_ids.device
    model.eval()

    B, T = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # Init cache and model position
    cache = KVCache(n_layers=model.cfg["n_layers"])

    # Prefill
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]

    # Decode
    cur_attn = attn_mask
    for step in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

        yield next_token
        if step + 1 == max_new_tokens:
            break

        # Extend mask to include the newly generated token
        if cur_attn is not None:
            ones = torch.ones((B, 1), dtype=cur_attn.dtype, device=device)
            cur_attn = torch.cat([cur_attn, ones], dim=1)

        # Advance one token with KV cache
        out = model(next_token, cache=cache, attn_mask=cur_attn)[:, -1]
        token_ids = torch.cat([token_ids, next_token], dim=1)


def shrink_kv_cache_inplace(cache, keep_mask, n_layers):
    keep_mask = keep_mask.bool()
    for i in range(n_layers):
        state = cache.get(i)
        if isinstance(state, tuple):
            cache.update(i, tuple(t[keep_mask] for t in state))
        elif state is not None:
            cache.update(i, state[keep_mask])



@torch.inference_mode()
def generate_text_basic_batched_cache_stop(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    """
    Same as generate_text_basic_batched_cache but
    with per-sequence early stop.
    I.e., finished rows that see an EOS written don't
    participate in forward pass anymore.
    """
    device = token_ids.device
    model.eval()

    B, T0 = token_ids.shape

    # Build attention mask
    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # Init cache and prefill once on full batch
    cache = KVCache(n_layers=model.cfg["n_layers"])
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]  # (B, V)

    finished_full = torch.zeros(B, dtype=torch.bool, device=device)
    active_idx = torch.arange(B, device=device)  # active rows -> original rows
    cur_attn_active = attn_mask                  # mirrors the active cache
    generated_full_steps = []                    # list of (B,1) step tensors

    for step in range(max_new_tokens):
        # Next tokens for the active sub-batch
        next_token_active = torch.argmax(out, dim=-1, keepdim=True)  # (B_active, 1)

        # Scatter into a full-sized (B,1) step tensor (EOS for finished rows)
        fill_val = int(eos_token_id) if eos_token_id is not None else 0
        step_full = torch.full((B, 1), fill_value=fill_val,
                               dtype=token_ids.dtype, device=device)
        step_full.index_copy_(0, active_idx, next_token_active)
        generated_full_steps.append(step_full)
        if step + 1 == max_new_tokens:
            break

        # Update finished bookkeeping in full-batch coordinates
        if eos_token_id is not None:
            newly_finished_active = (next_token_active.squeeze(1) == eos_token_id)
            finished_full.index_put_(
                (active_idx,),
                newly_finished_active | finished_full.index_select(0, active_idx)
            )
        else:
            newly_finished_active = torch.zeros_like(
                next_token_active.squeeze(1), dtype=torch.bool, device=device
            )

        if eos_token_id is not None and torch.all(finished_full):
            break

        # Keep only survivors in the compute batch
        keep_mask_active = ~newly_finished_active
        if keep_mask_active.ndim == 0:
            keep_any = bool(keep_mask_active.item())
        else:
            keep_any = bool(keep_mask_active.any().item())
        if not keep_any:
            break

        next_token_survivors = next_token_active[keep_mask_active]  # (B_surv, 1)
        active_idx = active_idx[keep_mask_active]

        # Shrink attn mask and append a "1" for the generated token
        if cur_attn_active is not None:
            cur_attn_active = cur_attn_active[keep_mask_active]
            ones = torch.ones((cur_attn_active.size(0), 1),
                              dtype=cur_attn_active.dtype, device=device)
            cur_attn_active = torch.cat([cur_attn_active, ones], dim=1)

        # Shrink KV cache along batch dim to survivors
        shrink_kv_cache_inplace(cache, keep_mask_active, model.cfg["n_layers"])

        # Advance one token for survivors only
        out = model(next_token_survivors, cache=cache, attn_mask=cur_attn_active)[:, -1]

    # Concatenate per-step tensors; return only the generated part
    if generated_full_steps:
        return torch.cat(generated_full_steps, dim=1)  # (B, L_generated)
    else:
        return torch.empty((B, 0), dtype=token_ids.dtype, device=device)


@torch.inference_mode()
def generate_text_basic_batched_stream_cache_stop(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    attn_mask: torch.Tensor | None = None,
    pad_id: int | None = None,
):
    """
    Same as generate_text_basic_batched_stream_cache but
    with per-sequence early stop.
    """
    device = token_ids.device
    model.eval()

    B, T0 = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]  # (B, V)

    finished_full = torch.zeros(B, dtype=torch.bool, device=device)
    active_idx = torch.arange(B, device=device)
    cur_attn_active = attn_mask

    for step in range(max_new_tokens):
        next_token_active = torch.argmax(out, dim=-1, keepdim=True)  # (B_active, 1)

        # Build full-sized step to yield
        fill_val = int(eos_token_id) if eos_token_id is not None else 0
        step_full = torch.full((B, 1), fill_value=fill_val,
                               dtype=token_ids.dtype, device=device)
        step_full.index_copy_(0, active_idx, next_token_active)

        if eos_token_id is not None:
            newly_finished_active = (next_token_active.squeeze(1) == eos_token_id)
            finished_full.index_put_(
                (active_idx,),
                newly_finished_active | finished_full.index_select(0, active_idx)
            )
        else:
            newly_finished_active = torch.zeros_like(
                next_token_active.squeeze(1), dtype=torch.bool, device=device
            )

        # Yield before shrinking so callers still see exactly one (B,1) per step
        yield step_full
        if step + 1 == max_new_tokens:
            break

        if eos_token_id is not None and torch.all(finished_full):
            break

        keep_mask_active = ~newly_finished_active
        if keep_mask_active.ndim == 0:
            keep_any = bool(keep_mask_active.item())
        else:
            keep_any = bool(keep_mask_active.any().item())
        if not keep_any:
            break

        next_token_survivors = next_token_active[keep_mask_active]
        active_idx = active_idx[keep_mask_active]

        if cur_attn_active is not None:
            cur_attn_active = cur_attn_active[keep_mask_active]
            ones = torch.ones((cur_attn_active.size(0), 1),
                              dtype=cur_attn_active.dtype, device=device)
            cur_attn_active = torch.cat([cur_attn_active, ones], dim=1)

        shrink_kv_cache_inplace(cache, keep_mask_active, model.cfg["n_layers"])

        out = model(next_token_survivors, cache=cache, attn_mask=cur_attn_active)[:, -1]


def load_model_and_tokenizer(which_model="base", device="cpu", use_compile=False,
                             local_dir="lfm3", float32_upcast=True, checkpoint_path=None):
    # SDPA handles attention accumulation; retain the chapter script argument.
    return _load_model_and_tokenizer(which_model, device, use_compile, local_dir, checkpoint_path)

