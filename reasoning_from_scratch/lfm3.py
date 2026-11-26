"""LFM3 in PyTorch, with the tensor interface used by the chapter algorithms.

Architecture and weight names follow LiquidAI/LFM3-1.7B-Base. The reference
modeling code is Apache-2.0, Copyright 2026 The HuggingFace Team:
https://huggingface.co/LiquidAI/LFM3-1.7B-Base/tree/main
The pretrained weights are separately covered by the LFM Open License v1.0.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


MODEL_ID = "LiquidAI/LFM3-1.7B-Base"
MODEL_REVISION = "7bca5038823c0d0f14efe06439109354a10e014c"
# LFM3-specific: the pretrained candidate mixes convolution, sliding, and
# global attention layers; a Qwen configuration cannot load these weights.
LFM3_CONFIG_1_7B = {
    "vocab_size": 128_000,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 16,
    "n_layers": 24,
    "hidden_dim": 7680,
    "head_dim": 128,
    "n_kv_groups": 4,
    "conv_kernel_size": 7,
    "sliding_window": 256,
    "norm_eps": 1e-5,
    "dtype": torch.bfloat16,
    "layer_types": (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
    ),
}


class KVCache:
    """Store attention (K, V) or a convolution input window per layer."""

    def __init__(self, n_layers):
        # LFM3-specific: convolution windows and attention K/V share a cache.
        self.cache = [None] * n_layers
        self.position = 0

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        self.cache = [None] * len(self.cache)
        self.position = 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, hidden, dtype = cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"]
        self.w1 = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        self.w2 = nn.Linear(hidden, dim, bias=False, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ShortConv(nn.Module):
    # LFM3-specific: gated causal depthwise convolution replaces attention
    # in 13 of the candidate's 24 layers.
    def __init__(self, cfg):
        super().__init__()
        dim, dtype = cfg["emb_dim"], cfg["dtype"]
        self.width = cfg["conv_kernel_size"]
        self.conv = nn.Conv1d(
            dim, dim, self.width, groups=dim, bias=False, dtype=dtype
        )
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)

    def forward(self, x, state=None, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        b, c, values = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
        values = b * values
        if state is None:
            history = F.pad(values, (self.width - 1, 0))
        else:
            history = torch.cat((state, values), dim=-1)
        out = self.conv(history)
        next_state = (
            history[..., -(self.width - 1) :]
            if self.width > 1
            else history[..., :0]
        )
        return self.out_proj((c * out).transpose(1, 2)), next_state


class Attention(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        dim, dtype = cfg["emb_dim"], cfg["dtype"]
        self.n_heads = cfg["n_heads"]
        self.n_kv = cfg["n_kv_groups"]
        self.head_dim = cfg["head_dim"]
        self.window = (
            cfg["sliding_window"]
            if layer_type == "sliding_attention"
            else None
        )
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv * self.head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv * self.head_dim, bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=False, dtype=dtype
        )
        self.gate_proj = nn.Linear(
            dim, self.n_heads, bias=False, dtype=dtype
        )
        self.q_norm = RMSNorm(self.head_dim, cfg["norm_eps"], dtype)
        self.k_norm = RMSNorm(self.head_dim, cfg["norm_eps"], dtype)
        # LFM3-specific: global layers rotate half the dimensions, retaining
        # the full-head
        # frequency progression; the remaining frequencies are exactly zero.
        rotated = self.head_dim if self.window else self.head_dim // 2
        base = 10_000.0 if self.window else 1_000_000.0
        self.rotated_dim = rotated
        self.rope_base = base

    def _rotate(self, x, positions):
        # Compute in FP32 on the actual device, including after HF meta loading
        # or dtype conversion. No uninitialized nonpersistent buffers survive.
        freq = self.rope_base ** (
            -torch.arange(
                0, self.rotated_dim, 2, device=x.device, dtype=torch.float32
            )
            / self.head_dim
        )
        freq = F.pad(freq, (0, self.head_dim // 2 - freq.numel()))
        angles = positions.float().unsqueeze(-1) * freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos, sin = (
            angles.cos().to(x.dtype).float(),
            angles.sin().to(x.dtype).float(),
        )
        half = self.head_dim // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (x.float() * cos + rotated.float() * sin).to(x.dtype)

    def forward(self, x, positions, start, state=None, mask=None):
        batch, length, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(x).view(batch, length, self.n_kv, self.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(x)
            .view(batch, length, self.n_kv, self.head_dim)
            .transpose(1, 2)
        )
        q, k = self._rotate(q, positions), self._rotate(k, positions)
        if state is not None:
            k, v = (
                torch.cat((state[0], k), dim=2),
                torch.cat((state[1], v), dim=2),
            )
        next_state = (k, v)
        if self.window is not None:
            next_state = (k[:, :, -self.window :], v[:, :, -self.window :])
        # Cached sliding layers keep a suffix; absolute positions still advance.
        key_start = start + length - k.shape[2]
        query_pos = torch.arange(start, start + length, device=x.device)
        key_pos = torch.arange(key_start, start + length, device=x.device)
        allowed = key_pos[None, :] <= query_pos[:, None]
        if self.window is not None:
            allowed = allowed & (
                key_pos[None, :] > query_pos[:, None] - self.window
            )
        allowed = allowed[None, None]
        if mask is not None:
            allowed = (
                allowed
                & mask[:, None, None, key_start : start + length].bool()
            )
        k, v = (
            k.repeat_interleave(self.n_heads // self.n_kv, 1),
            v.repeat_interleave(self.n_heads // self.n_kv, 1),
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        gate = self.gate_proj(x).sigmoid().unsqueeze(-1)
        out = (out.transpose(1, 2) * gate).reshape(batch, length, -1)
        return self.o_proj(out), next_state


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "conv":
            self.conv = ShortConv(cfg)
        else:
            self.self_attn = Attention(cfg, layer_type)
        self.feed_forward = FeedForward(cfg)
        for name in (
            "operator_norm",
            "ffn_norm",
            "post_operator_norm",
            "post_ffn_norm",
        ):
            setattr(
                self,
                name,
                RMSNorm(cfg["emb_dim"], cfg["norm_eps"], cfg["dtype"]),
            )

    def forward(self, x, positions, start, state=None, mask=None):
        normalized = self.operator_norm(x)
        if self.layer_type == "conv":
            current_mask = (
                mask[:, -x.shape[1] :] if mask is not None else None
            )
            out, state = self.conv(normalized, state, current_mask)
        else:
            out, state = self.self_attn(
                normalized, positions, start, state, mask
            )
        x = x + self.post_operator_norm(out)
        return x + self.post_ffn_norm(
            self.feed_forward(self.ffn_norm(x))
        ), state


class LFM3Model(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = dict(LFM3_CONFIG_1_7B if cfg is None else cfg)
        cfg = self.cfg
        if len(cfg["layer_types"]) != cfg["n_layers"]:
            raise ValueError("layer_types must contain one entry per layer")
        if cfg["n_heads"] % cfg["n_kv_groups"] or cfg["head_dim"] % 4:
            raise ValueError("Invalid attention head configuration")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.model.layers = nn.ModuleList(
            [DecoderLayer(cfg, kind) for kind in cfg["layer_types"]]
        )
        self.model.embedding_norm = RMSNorm(
            cfg["emb_dim"], cfg["norm_eps"], cfg["dtype"]
        )
        self.lm_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
            dtype=cfg["dtype"],
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def tok_emb(self):
        return self.model.embed_tokens

    @property
    def trf_blocks(self):
        return self.model.layers

    def reset_kv_cache(self):
        # Positions and recurrent state belong to the cache, not the model.
        pass

    def forward(self, in_idx, cache=None, attn_mask=None):
        if in_idx.ndim != 2 or not all(in_idx.shape):
            raise ValueError(
                "in_idx must be a nonempty [batch, tokens] tensor"
            )
        start = cache.position if cache is not None else 0
        end = start + in_idx.shape[1]
        if end > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {end} exceeds context length "
                f"{self.cfg['context_length']}"
            )
        if attn_mask is not None and attn_mask.shape != (
            in_idx.shape[0],
            end,
        ):
            raise ValueError(
                "attn_mask must cover the entire cached sequence"
            )
        x = self.model.embed_tokens(in_idx)
        positions = torch.arange(
            start, end, device=in_idx.device
        ).unsqueeze(0)
        if attn_mask is not None:
            positions = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)[
                :, start:end
            ]
        for i, layer in enumerate(self.model.layers):
            state = cache.get(i) if cache is not None else None
            x, state = layer(x, positions, start, state, attn_mask)
            if cache is not None:
                cache.update(i, state)
        if cache is not None:
            cache.position = end
        return self.lm_head(self.model.embedding_norm(x))


class LFM3Tokenizer:
    """Chapter tokenizer API using LFM3's vocabulary and text chat format.

    Chat formatting covers text system/user/assistant messages. Raw completion
    encoding adds no BOS/EOS; training code explicitly appends EOS to targets.
    """

    def __init__(
        self,
        tokenizer_file_path="lfm3/tokenizer.json",
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False,
    ):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_file_path))
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|im_end|>"
        self.pad_token = "<|pad|>"
        self.bos_token_id = self.token_to_id(self.bos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.pad_token_id = self.token_to_id(self.pad_token)
        self._special_to_id = {
            t: self.token_to_id(t)
            for t in (
                self.bos_token,
                self.eos_token,
                self.pad_token,
                "<|im_start|>",
                "<|endoftext|>",
                "<think>",
                "</think>",
            )
        }

    def token_to_id(self, token):
        result = self._tok.token_to_id(token)
        if result is None:
            raise ValueError(f"Required LFM3 token missing: {token}")
        return result

    def format_messages(self, messages, add_generation_prompt=True):
        # LFM3-specific: one BOS, im_start/im_end, and think delimiters.
        text = self.bos_token
        for message in messages:
            role, content = message["role"], message["content"]
            if role not in (
                "system",
                "user",
                "assistant",
            ) or not isinstance(content, str):
                raise ValueError(
                    "Only text system/user/assistant messages are supported"
                )
            if role == "assistant" and not content.startswith("<think>"):
                content = "<think></think>" + content
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            text += "" if self.add_thinking else "<think></think>"
        return text

    def encode(self, prompt, chat_wrapped=None):
        wrapped = (
            self.apply_chat_template
            if chat_wrapped is None
            else chat_wrapped
        )
        if prompt in self._special_to_id:
            return [self._special_to_id[prompt]]
        if wrapped:
            prompt = self.format_messages(
                [{"role": "user", "content": prompt}],
                self.add_generation_prompt,
            )
        return self._tok.encode(prompt, add_special_tokens=False).ids

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._tok.decode(token_ids, skip_special_tokens=False)


def download_lfm3(
    kind="base", tokenizer_only=False, out_dir="lfm3", include_code=False
):
    """Download the pinned base weights; 'reasoning' selects chat prompting only."""
    from huggingface_hub import snapshot_download

    if kind not in ("base", "reasoning"):
        raise ValueError("kind must be 'base' or 'reasoning'")
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
        "LICENSE",
    ]
    if not tokenizer_only:
        files.append("model.safetensors")
    if include_code:
        files.extend(
            ["configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"]
        )
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=out_dir,
            allow_patterns=files,
        )
    )


def load_checkpoint(model, path):
    """Strictly load native LFM3 safetensors or a locally trained state dict."""
    model.load_state_dict(load_state_dict_file(path), strict=True)
    return model


def load_pretrained_model(path, device="cpu", cfg=None, dtype=None):
    """Load weights without first allocating a random 1.7B-parameter model."""
    # LFM3-specific: source weights are BF16. CPU/MPS default to FP32 for
    # stable cache comparisons and small optimizer updates. Accelerators with
    # native BF16 kernels retain BF16; callers can choose dtype explicitly.
    if dtype is None and torch.device(device).type in ("cpu", "mps"):
        dtype = torch.float32
    # LFM3-specific: native names match Liquid AI's safetensors directly.
    with torch.device("meta"):
        model = LFM3Model(cfg)
    model.load_state_dict(load_state_dict_file(path), strict=True, assign=True)
    # assign=True creates separate Parameter objects for the shared tensors.
    model.lm_head.weight = model.model.embed_tokens.weight
    model = model.to(device=device, dtype=dtype)
    model.cfg["dtype"] = next(model.parameters()).dtype
    return model.eval()


def load_state_dict_file(path, **kwargs):
    """Read either upstream safetensors or a chapter training checkpoint."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
        state.setdefault(
            "lm_head.weight", state["model.embed_tokens.weight"]
        )
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    return state


def load_tokenizer_only(which_model="base", local_dir="lfm3"):
    download_lfm3(which_model, tokenizer_only=True, out_dir=local_dir)
    return LFM3Tokenizer(
        Path(local_dir) / "tokenizer.json",
        apply_chat_template=which_model == "reasoning",
        add_generation_prompt=which_model == "reasoning",
        add_thinking=which_model == "reasoning",
    )


def load_model_and_tokenizer(
    which_model="base",
    device="cpu",
    use_compile=False,
    local_dir="lfm3",
    checkpoint_path=None,
    dtype=None,
):
    tokenizer = load_tokenizer_only(which_model, local_dir)
    if checkpoint_path is None:
        download_lfm3(which_model, out_dir=local_dir)
        checkpoint_path = Path(local_dir) / "model.safetensors"
    model = load_pretrained_model(checkpoint_path, device=device, dtype=dtype)
    if use_compile:
        model = torch.compile(model)
    return model, tokenizer


def export_to_huggingface(
    model, output_dir, tokenizer_dir="lfm3", tokenizer_kind="base"
):
    """Save the trained model in the original Liquid AI safetensors layout."""
    import shutil
    from safetensors.torch import save_file

    if tokenizer_kind not in ("base", "reasoning"):
        raise ValueError("tokenizer_kind must be 'base' or 'reasoning'")
    source = download_lfm3(
        tokenizer_only=True, out_dir=tokenizer_dir, include_code=True
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if k != "lm_head.weight"
    }
    save_file(
        state, str(output / "model.safetensors"), metadata={"format": "pt"}
    )
    config = json.loads((source / "config.json").read_text())
    mapping = {
        "vocab_size": "vocab_size",
        "emb_dim": "hidden_size",
        "hidden_dim": "intermediate_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "n_kv_groups": "num_key_value_heads",
        "head_dim": "head_dim",
        "conv_kernel_size": "conv_L_cache",
        "sliding_window": "sliding_window",
        "context_length": "max_position_embeddings",
        "norm_eps": "norm_eps",
        "layer_types": "layer_types",
    }
    config.update(
        {target: model.cfg[key] for key, target in mapping.items()}
    )
    config.update(
        dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
        tokenizer_kind=tokenizer_kind,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
        "LICENSE",
        "configuration_lfm3.py",
        "modeling_lfm3.py",
        "lfm3_conv.py",
    ):
        if (source / name).resolve() != (output / name).resolve():
            shutil.copyfile(source / name, output / name)
    (output / "base_model.json").write_text(
        json.dumps(
            {"model_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2
        )
    )
    return output
