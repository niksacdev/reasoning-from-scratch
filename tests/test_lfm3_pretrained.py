"""Opt-in, full-weight integration checks (no substitute random weights).

LFM3_PRETRAINED_DIR=/path/to/lfm3 LFM3_TEST_DEVICE=mps pytest -s
    tests/test_lfm3_pretrained.py

Weights are only read. Temporary training checkpoints are deleted after the
round trip. This is a bounded correctness check, not a model-quality score.
"""

import gc
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from reasoning_from_scratch import ch02, ch04, ch05, ch06, ch08
from reasoning_from_scratch import lfm3, lfm3_batched


pytestmark = pytest.mark.skipif(
    not os.environ.get("LFM3_PRETRAINED_DIR"),
    reason="Set LFM3_PRETRAINED_DIR to the actual pinned pretrained weights",
)


def emit_result(name, result):
    print(json.dumps({"check": name, **result}), flush=True)
    report = os.environ.get("LFM3_VALIDATION_REPORT")
    if report:
        with Path(report).open("a") as stream:
            stream.write(json.dumps({"check": name, **result}) + "\n")


@pytest.fixture
def candidate():
    source = Path(os.environ["LFM3_PRETRAINED_DIR"])
    device = os.environ.get("LFM3_TEST_DEVICE", "cpu")
    model = lfm3.load_pretrained_model(source / "model.safetensors", device=device)
    tokenizer = lfm3.LFM3Tokenizer(source / "tokenizer.json")
    yield model, tokenizer, source, device
    del model
    gc.collect()


def test_actual_weights_cache_benchmark_and_decisions(candidate):
    model, tokenizer, _, device = candidate
    ids = torch.tensor([tokenizer.encode("The nematode C. elegans is")], device=device)
    result = ch02.benchmark_generation(model, ids, max_new_tokens=32, repeats=3)
    assert result["identical_tokens"]
    output = ch02.generate_text_basic_cache(model, ids, 32)
    result["text"] = tokenizer.decode(output[0].tolist())
    result["token_ids"] = output[0].tolist()
    emit_result("pretrained_cache", result)


def test_actual_weights_batched_and_sampling_paths(candidate):
    model, tokenizer, _, device = candidate
    prompts = ["The capital of Germany is", "2 + 2 ="]
    encoded = [tokenizer.encode(p) for p in prompts]
    size = max(map(len, encoded))
    ids = torch.tensor([[tokenizer.pad_token_id] * (size - len(row)) + row
                        for row in encoded], device=device)
    expected = torch.cat([ch02.generate_text_basic_cache(
        model, torch.tensor([row], device=device), 4
    ) for row in encoded])
    batched = lfm3_batched.generate_text_basic_batched_cache_stop(
        model, ids, 4, pad_id=tokenizer.pad_token_id
    )
    torch.testing.assert_close(batched, expected)
    torch.manual_seed(123)
    sampled, length, text = ch06.sample_response(
        model, tokenizer, prompts[0], device, max_new_tokens=4,
        temperature=0.8, top_p=0.9,
    )
    assert sampled.numel() > length
    score = ch05.avg_logprob_answer(model, tokenizer, prompts[0], " Berlin.", device)
    assert torch.isfinite(torch.tensor(score))
    vote = ch04.self_consistency_vote(
        model=model, tokenizer=tokenizer, prompt="What is 1+1?",
        device=device, num_samples=2, max_new_tokens=4,
    )
    emit_result("pretrained_sampling", {
        "batched_text": [tokenizer.decode(row.tolist()) for row in batched],
        "sampled_text": text, "answer_logprob": score,
        "vote": str(vote),
    })


@pytest.mark.skipif(not os.environ.get("LFM3_REFERENCE_DIR"),
                    reason="Set LFM3_REFERENCE_DIR to the pinned official code")
def test_actual_weights_match_official_reference(candidate, tmp_path):
    from transformers import AutoModelForCausalLM

    model, tokenizer, source, device = candidate
    reference = Path(os.environ["LFM3_REFERENCE_DIR"])
    checkpoint = tmp_path / "reference"
    checkpoint.mkdir()
    for name in ("config.json", "generation_config.json", "configuration_lfm3.py",
                 "modeling_lfm3.py", "lfm3_conv.py"):
        shutil.copyfile(reference / name, checkpoint / name)
    # Reference from_pretrained also materializes nonpersistent RoPE buffers;
    # assigning only a state dict to a meta model would leave those on meta.
    weights = checkpoint / "model.safetensors"
    try:
        weights.hardlink_to(source / "model.safetensors")
    except OSError:
        weights.symlink_to((source / "model.safetensors").resolve())
    official = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, attn_implementation="eager",
        dtype=next(model.parameters()).dtype,
    ).to(device).eval()
    ids = torch.tensor([tokenizer.encode("The nematode C. elegans is")], device=device)
    with torch.inference_mode():
        expected = official(ids, use_cache=False).logits
        actual = model(ids)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        reference_ids = official.generate(ids, max_new_tokens=32, do_sample=False, use_cache=True)
        native_ids = ch02.generate_text_basic_cache(model, ids, 32)
        torch.testing.assert_close(reference_ids[:, ids.shape[1]:], native_ids)
    emit_result("pretrained_reference", {
        "max_logit_difference": (actual - expected).abs().max().item(),
        "identical_32_tokens": True,
    })
    weights.unlink()


def test_actual_weights_backward_optimizer_checkpoint(candidate, tmp_path):
    model, tokenizer, _, device = candidate
    prompt = tokenizer.encode("2 + 2 =")
    answer = tokenizer.encode(" 4.", chat_wrapped=False)
    example = {"token_ids": prompt + answer + [tokenizer.eos_token_id],
               "prompt_len": len(prompt)}
    model.train()
    parameter = model.model.layers[0].conv.in_proj.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    loss = ch08.compute_example_loss(model, example, device)
    loss.backward()
    assert torch.isfinite(loss)
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    optimizer.step()
    changed = (parameter.detach() != before).sum().item()
    assert changed > 0
    optimizer.zero_grad(set_to_none=True)
    del optimizer, before
    model.eval()
    ids = torch.tensor([prompt], device=device)
    with torch.inference_mode():
        expected = model(ids).cpu()
    checkpoint = ch08.save_checkpoint(model, tmp_path, 1)
    restored = lfm3.load_pretrained_model(checkpoint, device=device)
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids).cpu(), expected)
    checkpoint.unlink()
    emit_result("pretrained_training", {
        "loss": loss.item(), "changed_first_conv_weights": changed,
        "checkpoint_roundtrip": True, "dtype": str(parameter.dtype),
    })


@pytest.mark.skipif(not os.environ.get("LFM3_REFERENCE_DIR"),
                    reason="Set LFM3_REFERENCE_DIR to the pinned official code")
def test_full_trained_export_roundtrip(candidate, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    model, tokenizer, _, device = candidate
    reference = Path(os.environ["LFM3_REFERENCE_DIR"])
    monkeypatch.setattr(lfm3, "download_lfm3", lambda **kwargs: reference)
    # Independently fine-tune one convolution parameter before exporting.
    # Do not depend on another test having modified a shared fixture.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter = model.model.layers[0].conv.in_proj.weight
    parameter.requires_grad_(True)
    prompt = tokenizer.encode("2 + 2 =")
    answer = tokenizer.encode(" 4.", chat_wrapped=False)
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    loss = ch08.compute_example_loss(
        model, {"token_ids": prompt + answer, "prompt_len": len(prompt)}, device
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ids = torch.tensor([tokenizer.encode("The capital of Germany is")], device=device)
    with torch.inference_mode():
        expected = model(ids).cpu()
    output = lfm3.export_to_huggingface(model, tmp_path / "hf-export")
    official = AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True, dtype=next(model.parameters()).dtype,
        attn_implementation="eager",
    ).to(device).eval()
    with torch.inference_mode():
        actual = official(ids, use_cache=False).logits.cpu()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    (output / "model.safetensors").unlink()
    emit_result("pretrained_trained_export", {
        "max_logit_difference": (actual - expected).abs().max().item(),
        "roundtrip": True, "finetuning_loss": loss.item(),
    })
