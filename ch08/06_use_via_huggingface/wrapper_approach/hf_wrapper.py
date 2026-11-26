from reasoning_from_scratch.lfm3 import load_pretrained_model
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path

import torch
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from reasoning_from_scratch.lfm3 import (
    LFM3_CONFIG_1_7B,
    LFM3Model,
    LFM3Tokenizer,
    download_lfm3,
)


MODEL_FILENAMES = {
    "base": "model.safetensors",
    "reasoning": "model.safetensors",
}

TOKENIZER_FILENAMES = {
    "base": "tokenizer.json",
    "reasoning": "tokenizer.json",
}


def _dtype_to_name(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        return str(dtype_value).removeprefix("torch.")
    return dtype_value


def _lfm3_cfg_to_jsonable(model_cfg):
    model_cfg = dict(model_cfg)
    model_cfg["dtype"] = _dtype_to_name(model_cfg["dtype"])
    return model_cfg


def _lfm3_cfg_from_jsonable(model_cfg):
    model_cfg = dict(model_cfg)
    if isinstance(model_cfg["dtype"], str):
        model_cfg["dtype"] = getattr(torch, model_cfg["dtype"])
    return model_cfg


class LFM3LocalWrapperConfig(PretrainedConfig):
    model_type = "lfm3_local_wrapper"

    def __init__(
        self,
        lfm3_cfg=None,
        tokenizer_kind="base",
        pad_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        if lfm3_cfg is None:
            lfm3_cfg = dict(LFM3_CONFIG_1_7B)

        self.lfm3_cfg = _lfm3_cfg_to_jsonable(lfm3_cfg)
        self.tokenizer_kind = tokenizer_kind
        self.vocab_size = lfm3_cfg["vocab_size"]
        self.hidden_size = lfm3_cfg["emb_dim"]
        self.num_attention_heads = lfm3_cfg["n_heads"]
        self.num_key_value_heads = lfm3_cfg["n_kv_groups"]
        self.num_hidden_layers = lfm3_cfg["n_layers"]
        self.max_position_embeddings = lfm3_cfg["context_length"]
        kwargs.pop("tie_word_embeddings", None)
        kwargs.pop("dtype", None)
        kwargs.pop("use_cache", None)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dtype=self.lfm3_cfg["dtype"],
            use_cache=False,
            tie_word_embeddings=True,
            **kwargs,
        )


class LFM3LocalWrapperForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"model.lm_head.weight": "model.model.embed_tokens.weight"}
    config_class = LFM3LocalWrapperConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.model = LFM3Model(_lfm3_cfg_from_jsonable(config.lfm3_cfg))
        self.generation_config.pad_token_id = config.pad_token_id
        self.generation_config.eos_token_id = config.eos_token_id
        self.generation_config.use_cache = False
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.model.embed_tokens = value
        self.model.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        del use_cache, kwargs
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        logits = self.model(input_ids, attn_mask=attention_mask)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict is False:
            outputs = (logits,)
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        del kwargs
        # This wrapper keeps things simple and recomputes from the full prefix
        # during generation instead of adapting the scratch KV cache to HF's
        # cache classes.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}


def build_tokenizer(tokenizer_kind="base", tokenizer_path=None, local_dir="lfm3"):
    if tokenizer_path is None:
        download_lfm3(
            kind=tokenizer_kind,
            tokenizer_only=True,
            out_dir=local_dir,
        )
        tokenizer_path = Path(local_dir) / TOKENIZER_FILENAMES[tokenizer_kind]
    else:
        tokenizer_path = Path(tokenizer_path)

    if tokenizer_kind == "reasoning":
        return LFM3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    return LFM3Tokenizer(tokenizer_file_path=tokenizer_path)


def build_model(
    tokenizer_kind="base",
    model_path=None,
    tokenizer=None,
    local_dir="lfm3",
):
    if tokenizer is None:
        tokenizer = build_tokenizer(
            tokenizer_kind=tokenizer_kind,
            local_dir=local_dir,
        )

    if model_path is None:
        download_lfm3(
            kind=tokenizer_kind,
            tokenizer_only=False,
            out_dir=local_dir,
        )
        model_path = Path(local_dir) / MODEL_FILENAMES[tokenizer_kind]
    else:
        model_path = Path(model_path)

    # LFM3-specific: share the strict native loader and its device/dtype rules.
    native = load_pretrained_model(model_path, cfg=LFM3_CONFIG_1_7B)
    config = LFM3LocalWrapperConfig(
        lfm3_cfg=native.cfg,
        tokenizer_kind=tokenizer_kind,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.device("meta"):
        model = LFM3LocalWrapperForCausalLM(config)
    model.model = native
    return model


def build_model_and_tokenizer(
    tokenizer_kind="base",
    model_path=None,
    tokenizer_path=None,
    local_dir="lfm3",
):
    tokenizer = build_tokenizer(
        tokenizer_kind=tokenizer_kind,
        tokenizer_path=tokenizer_path,
        local_dir=local_dir,
    )
    model = build_model(
        tokenizer_kind=tokenizer_kind,
        model_path=model_path,
        tokenizer=tokenizer,
        local_dir=local_dir,
    )
    return model, tokenizer


def format_distilled_answer(entry, use_think_tokens=False):
    content = str(entry["message_content"]).replace("<think>", "").replace("</think>", "").strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")

    thinking = str(entry.get("message_thinking", "")).replace("<think>", "").replace("</think>", "").strip()

    if use_think_tokens:
        return f"<think>{thinking}</think>\n\n{content}"

    if thinking:
        return f"{thinking}\n\n{content}"

    return content
