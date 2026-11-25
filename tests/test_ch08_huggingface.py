"""Verify the local LFM3 Hugging Face interface without downloading weights."""

import importlib.util
from pathlib import Path
import sys

import torch
import pytest

pytest.importorskip("transformers")

from reasoning_from_scratch.lfm3 import LFM3_CONFIG_1_7B, LFM3Model


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "ch08/06_use_via_huggingface/wrapper_approach/hf_wrapper.py"
spec = importlib.util.spec_from_file_location("lfm3_hf_wrapper", WRAPPER)
wrapper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wrapper
spec.loader.exec_module(wrapper)


def test_wrapper_preserves_padding_loss_and_generation(tmp_path):
    cfg = dict(LFM3_CONFIG_1_7B, vocab_size=32, emb_dim=16, hidden_dim=32,
               n_heads=2, n_kv_groups=1, head_dim=8, n_layers=3,
               layer_types=("conv", "sliding_attention", "full_attention"),
               dtype=torch.float32, sliding_window=4, context_length=32)
    config = wrapper.LFM3LocalWrapperConfig(lfm3_cfg=cfg, pad_token_id=0, eos_token_id=31)
    model = wrapper.LFM3LocalWrapperForCausalLM(config).eval()
    reference = LFM3Model(cfg).eval()
    reference.load_state_dict(model.model.state_dict())
    ids = torch.tensor([[0, 0, 2, 3, 4], [5, 6, 7, 8, 9]])
    mask = ids != 0
    outputs = model(ids, attention_mask=mask, labels=ids.masked_fill(~mask, -100))
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    torch.testing.assert_close(outputs.logits, reference(ids, attn_mask=mask))
    generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=2, do_sample=False)
    assert generated.shape == (2, 7)
    model.save_pretrained(tmp_path)
    restored = wrapper.LFM3LocalWrapperForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(restored(ids, attention_mask=mask).logits, outputs.logits)
    assert restored.model.lm_head.weight is restored.model.tok_emb.weight
