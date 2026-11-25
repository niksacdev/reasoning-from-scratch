"""Opt-in comparisons with the exact pinned Liquid AI reference implementation."""

import os
from pathlib import Path

import pytest
import torch

from reasoning_from_scratch.lfm3 import (
    KVCache, LFM3_CONFIG_1_7B, LFM3Model, LFM3Tokenizer, MODEL_ID, MODEL_REVISION,
    export_to_huggingface,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DOWNLOAD_TESTS") != "1",
    reason="Set RUN_REAL_DOWNLOAD_TESTS=1 to download pinned reference code and tokenizer",
)


@pytest.fixture
def reference_dir(tmp_path):
    supplied = os.environ.get("LFM3_REFERENCE_DIR")
    if supplied:
        return Path(supplied)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=tmp_path / "reference",
        allow_patterns=["*.py", "config.json", "tokenizer*.json", "chat_template.jinja",
                        "special_tokens_map.json", "generation_config.json", "LICENSE"],
    ))


def test_logits_gradients_cache_and_export_real_download(tmp_path, monkeypatch, reference_dir):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers import dynamic_module_utils
    from reasoning_from_scratch import lfm3

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    cfg = dict(LFM3_CONFIG_1_7B, emb_dim=16, hidden_dim=32, n_heads=2, n_kv_groups=1,
               head_dim=8, n_layers=3, layer_types=("conv", "sliding_attention", "full_attention"),
               dtype=torch.float32, sliding_window=4, context_length=64)
    hf_cfg = AutoConfig.from_pretrained(
        reference_dir, trust_remote_code=True, hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_hidden_layers=3,
        layer_types=list(cfg["layer_types"]), sliding_window=4,
        max_position_embeddings=64, dtype="float32",
    )
    reference = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True,
                                               attn_implementation="eager").float().eval()
    model = LFM3Model(cfg).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    expected, actual = reference(ids, use_cache=False).logits, model(ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    reference_parameters = dict(reference.named_parameters())
    for name, p in model.named_parameters():
        ref_p = reference_parameters[name]
        torch.testing.assert_close(p.grad, ref_p.grad, atol=1e-7, rtol=1e-4)
    cache = KVCache(cfg["n_layers"])
    cached = torch.cat([model(ids[:, i:i + 1], cache=cache) for i in range(ids.shape[1])], 1)
    torch.testing.assert_close(cached, expected, atol=1e-6, rtol=1e-5)
    monkeypatch.setattr(lfm3, "download_lfm3", lambda **kwargs: reference_dir)
    output = export_to_huggingface(model, tmp_path / "export", tokenizer_kind="reasoning")
    restored = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True).eval()
    torch.testing.assert_close(restored(ids, use_cache=False).logits, actual)
    tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    local = LFM3Tokenizer(reference_dir / "tokenizer.json")
    prompt = "The nematode C. elegans is"
    assert tokenizer(prompt, add_special_tokens=False).input_ids == local.encode(prompt)
    assert local.token_to_id("<think>") == 124901
    assert local.token_to_id("</think>") == 124902
    assert restored.generate(ids, max_new_tokens=2, do_sample=False).shape == (1, 10)


def test_full_candidate_parameter_layout_real_download(tmp_path, monkeypatch, reference_dir):
    from transformers import AutoConfig, AutoModelForCausalLM, dynamic_module_utils

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    config = AutoConfig.from_pretrained(reference_dir, trust_remote_code=True)
    with torch.device("meta"):
        reference = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        candidate = LFM3Model()
    expected = {key: tuple(value.shape) for key, value in reference.state_dict().items()}
    actual = {key: tuple(value.shape) for key, value in candidate.state_dict().items()}
    assert actual == expected
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in reference.parameters())
