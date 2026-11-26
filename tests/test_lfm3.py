"""Architecture, cache, padding and training regression tests for the candidate."""

from pathlib import Path
import os

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from reasoning_from_scratch.lfm3 import (
    KVCache, LFM3_CONFIG_1_7B, LFM3Model, LFM3Tokenizer, load_checkpoint,
)
from reasoning_from_scratch.lfm3_batched import shrink_kv_cache_inplace
from reasoning_from_scratch.ch06 import sequence_logprob
from reasoning_from_scratch.ch08 import build_examples, compute_example_loss


@pytest.fixture
def tiny_cfg():
    return dict(LFM3_CONFIG_1_7B, vocab_size=64, emb_dim=16, hidden_dim=32,
                n_heads=2, n_kv_groups=1, head_dim=8, n_layers=3,
                layer_types=("conv", "sliding_attention", "full_attention"),
                dtype=torch.float32, sliding_window=4, context_length=32)


@pytest.fixture
def tokenizer(tmp_path):
    specials = ("<|pad|>", "<|startoftext|>", "<|im_end|>", "<|im_start|>",
                "<|endoftext|>", "<think>", "</think>", "[UNK]")
    vocab = {token: i for i, token in enumerate(specials)}
    vocab.update({"answer": 8, "user": 9, "assistant": 10})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.add_special_tokens(list(specials))
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return LFM3Tokenizer(path, apply_chat_template=True, add_generation_prompt=True, add_thinking=True)


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1, 1, 1), (3, 2, 3), (7, 1)])
def test_cached_matches_full_sequence_across_sliding_window(tiny_cfg, chunks):
    torch.manual_seed(123)
    model = LFM3Model(tiny_cfg).eval()
    ids = torch.randint(0, 64, (2, sum(chunks)))
    expected = model(ids)
    cache = KVCache(tiny_cfg["n_layers"])
    outputs, start = [], 0
    for length in chunks:
        outputs.append(model(ids[:, start:start + length], cache=cache))
        start += length
    cached = torch.cat(outputs, 1)
    # CPU GEMV/GEMM kernels accumulate FP32 products differently when the
    # sequence is decoded token by token. Check both logits and decisions.
    torch.testing.assert_close(cached, expected, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(cached.argmax(-1), expected.argmax(-1))
    assert cache.get(0).shape[-1] == tiny_cfg["conv_kernel_size"] - 1
    assert cache.get(1)[0].shape[2] <= tiny_cfg["sliding_window"]
    cache.reset()
    assert cache.position == 0 and all(s is None for s in cache.get_all())


def test_left_padding_and_shrinking_preserve_both_state_types(tiny_cfg):
    torch.manual_seed(124)
    model = LFM3Model(tiny_cfg).eval()
    ids = torch.tensor([[0, 0, 2, 3, 4], [5, 6, 7, 8, 9]])
    mask = ids != 0
    cache = KVCache(3)
    logits = model(ids, cache=cache, attn_mask=mask)
    torch.testing.assert_close(logits[0, -3:], model(ids[:1, -3:])[0])
    shrink_kv_cache_inplace(cache, torch.tensor([True, False]), 3)
    next_mask = torch.tensor([[False, False, True, True, True, True]])
    next_logits = model(torch.tensor([[10]]), cache=cache, attn_mask=next_mask)
    torch.testing.assert_close(next_logits[:, -1], model(torch.tensor([[2, 3, 4, 10]]))[:, -1])


def test_grpo_and_distillation_backward_and_checkpoint(tiny_cfg, tmp_path):
    torch.manual_seed(125)
    model = LFM3Model(tiny_cfg)
    original = model.model.layers[0].conv.in_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([2, 3, 4, 5, 6])
    loss = compute_example_loss(model, {"token_ids": ids.tolist(), "prompt_len": 2}, "cpu")
    loss = loss - 0.01 * sequence_logprob(model, ids, prompt_len=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(original, model.model.layers[0].conv.in_proj.weight)
    checkpoint = tmp_path / "lfm3-1.7B-test.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = load_checkpoint(LFM3Model(tiny_cfg), checkpoint)
    torch.testing.assert_close(restored(ids.unsqueeze(0)), model(ids.unsqueeze(0)))
    assert restored.lm_head.weight is restored.tok_emb.weight


def test_tokenizer_and_distillation_use_candidate_tokens(tokenizer):
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert tokenizer.encode("<think>", chat_wrapped=False) == [tokenizer.token_to_id("<think>")]
    examples, skipped = build_examples([{
        "problem": "answer", "message_content": "answer", "message_thinking": "answer",
    }], tokenizer)
    assert skipped == 0
    tokens, prompt_len = examples[0]["token_ids"], examples[0]["prompt_len"]
    assert tokens[prompt_len] == tokenizer.token_to_id("<think>")
    assert tokens[-1] == tokenizer.eos_token_id
    assert tokenizer.encode("answer", chat_wrapped=False) == [8]


def test_context_limit_checked_before_cache_mutation(tiny_cfg):
    model = LFM3Model(tiny_cfg)
    cache = KVCache(3)
    with pytest.raises(ValueError, match="context length"):
        model(torch.ones(1, 33, dtype=torch.long), cache=cache)
    assert cache.position == 0 and cache.get(0) is None


def test_official_safetensors_layout_loads_without_conversion(tiny_cfg, tmp_path):
    from safetensors.torch import save_file

    model = LFM3Model(tiny_cfg)
    state = {key: value.clone() for key, value in model.state_dict().items() if key != "lm_head.weight"}
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, str(checkpoint))
    restored = load_checkpoint(LFM3Model(tiny_cfg), checkpoint)
    ids = torch.tensor([[3, 4, 5]])
    torch.testing.assert_close(restored(ids), model(ids))


def test_appendix_c_matches_shared_implementation(tiny_cfg):
    from conftest import import_definitions_from_notebook

    notebook = Path(__file__).resolve().parents[1] / "chC/01_main-chapter-code/chC_main.ipynb"
    module = import_definitions_from_notebook(notebook, "lfm3_appendix_defs")
    model = LFM3Model(tiny_cfg)
    appendix_model = module.LFM3Model(tiny_cfg)
    appendix_model.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(appendix_model(ids), model(ids))


@pytest.mark.skipif(os.environ.get("RUN_LFM3_COMPILE_TESTS") != "1",
                    reason="Opt-in: requires a working local C++ compiler")
def test_compiled_native_model_generation_with_and_without_cache(tiny_cfg):
    from reasoning_from_scratch.ch02 import generate_text_basic, generate_text_basic_cache

    torch.manual_seed(123)
    model = LFM3Model(tiny_cfg).eval()
    compiled = torch.compile(model)
    ids = torch.tensor([[2, 3, 4]])
    expected = generate_text_basic(model, ids, 3)
    torch.testing.assert_close(generate_text_basic(compiled, ids, 3), expected)
    torch.testing.assert_close(generate_text_basic_cache(compiled, ids, 3), expected)
