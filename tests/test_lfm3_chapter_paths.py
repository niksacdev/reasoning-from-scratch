"""Exercise chapter algorithms with the candidate's hybrid architecture.

Small dimensions keep CI inexpensive; pretrained checks live in
test_lfm3_pretrained.py. Fixtures control rewards, not model computations.
"""

import copy
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from reasoning_from_scratch import ch02, ch04, ch05, ch06, ch08, lfm3_batched
from reasoning_from_scratch.lfm3 import LFM3Model, load_pretrained_model
import test_lfm3 as lfm3_fixtures


tiny_cfg = lfm3_fixtures.tiny_cfg
tokenizer = lfm3_fixtures.tokenizer


ROOT = Path(__file__).resolve().parents[1]
GRPO_PATHS = [
    "reasoning_from_scratch/ch06.py",
    "ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py",
    "ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl_batched.py",
    "ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl_batched_fsdp.py",
    *[str(p.relative_to(ROOT)) for p in sorted(
        (ROOT / "ch07/03_rlvr_grpo_scripts_advanced").rglob("*.py")
    )],
]


def load_script(path):
    if path == "reasoning_from_scratch/ch06.py":
        return ch06
    spec = importlib.util.spec_from_file_location("chapter_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model(tiny_cfg):
    torch.manual_seed(701)
    return LFM3Model(tiny_cfg).eval()


GENERATORS = [
    ch02.generate_text_basic, ch02.generate_text_basic_cache,
    ch02.generate_text_basic_stream, ch02.generate_text_basic_stream_cache,
    ch04.generate_text_temp_stream_cache, ch04.generate_text_top_p_stream_cache,
    lfm3_batched.generate_text_basic_batched_cache,
    lfm3_batched.generate_text_basic_batched_stream_cache,
    lfm3_batched.generate_text_basic_batched_cache_stop,
    lfm3_batched.generate_text_basic_batched_stream_cache_stop,
]


def collect(generate, model, ids, budget, **kwargs):
    output = generate(model, ids, max_new_tokens=budget, **kwargs)
    if isinstance(output, torch.Tensor):
        return output
    steps = list(output)
    return torch.cat(steps, 1) if steps else ids[:, :0]


@pytest.mark.parametrize("generate", GENERATORS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("budget", [0, 1, 5])
def test_all_generation_paths_match_uncached_lfm3(model, generate, budget):
    ids = torch.tensor([[2, 3, 4]])
    calls = []
    hook = model.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[1]))
    actual = collect(generate, model, ids, budget)
    hook.remove()
    expected = ch02.generate_text_basic(model, ids, budget)
    torch.testing.assert_close(actual, expected)
    if budget == 0:
        assert calls == []
    elif "cache" in generate.__name__:
        assert calls == [3] + [1] * (budget - 1)


@pytest.mark.parametrize("generate", GENERATORS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_all_generation_paths_stop_on_eos_without_extra_forward(model, generate, batch_size):
    calls = []
    def force_eos(_, args, output):
        calls.append(args[0].shape[1])
        output.fill_(-1000)
        output[..., 0] = 1000
        return output
    model.register_forward_hook(force_eos)
    result = collect(generate, model, torch.tensor([[2, 3]] * batch_size), 8, eos_token_id=0)
    assert result.numel() <= batch_size and (result == 0).all()
    assert calls == [2]


@pytest.mark.parametrize("generate", GENERATORS[-4:], ids=lambda f: f.__name__)
def test_batched_left_padding_matches_individual_generations(model, generate):
    ids = torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]])
    actual = collect(generate, model, ids, 6, pad_id=0)
    expected = torch.cat([
        ch02.generate_text_basic(model, row[row != 0].unsqueeze(0), 6)
        for row in ids
    ])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("generate", GENERATORS[-2:], ids=lambda f: f.__name__)
def test_finished_rows_shrink_both_lfm3_state_types(model, generate):
    batch_sizes = []
    def force_different_eos_steps(_, args, output):
        batch_sizes.append(args[0].shape[0])
        output.fill_(-1000)
        output[..., 8] = 10
        if len(batch_sizes) == 1 or len(batch_sizes) == 3:
            output[0, -1, 0] = 20
        return output
    model.register_forward_hook(force_different_eos_steps)
    actual = collect(generate, model, torch.tensor([[2, 3], [4, 5]]), 8, eos_token_id=0)
    assert actual.tolist() == [[0, 0, 0], [8, 8, 0]]
    assert batch_sizes == [2, 1, 1]


def test_benchmark_checks_equal_work_and_tokens(model):
    result = ch02.benchmark_generation(model, torch.tensor([[2, 3]]), 4, repeats=2)
    assert result["identical_tokens"]
    assert result["speedup"] > 0
    for path in ("cached", "uncached"):
        assert len(result[path]["timings"]) == 2
        assert result[path]["tokens_per_second"] == 4 / result[path]["seconds"]


def test_meta_checkpoint_loader_preserves_tying_dtype_and_logits(model, tiny_cfg, tmp_path):
    path = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), path)
    loaded = load_pretrained_model(path, cfg=tiny_cfg)
    assert loaded.lm_head.weight is loaded.tok_emb.weight
    ids = torch.tensor([[2, 3, 4]])
    torch.testing.assert_close(loaded(ids), model(ids))
    assert loaded.cfg["dtype"] == next(loaded.parameters()).dtype


def test_answer_scoring_uses_raw_continuation(model, tokenizer):
    prompt, answer = "answer", "answer answer"
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer, chat_wrapped=False)
    ids = torch.tensor([prompt_ids + answer_ids])
    logp = model(ids).float().log_softmax(-1)
    expected = torch.stack([
        logp[0, len(prompt_ids) - 1 + i, t] for i, t in enumerate(answer_ids)
    ]).mean().item()
    assert ch05.avg_logprob_answer(model, tokenizer, prompt, answer) == pytest.approx(expected)
    with pytest.raises(ValueError, match="Prompt and answer"):
        ch05.avg_logprob_answer(model, tokenizer, prompt, "")


@pytest.mark.parametrize("script", GRPO_PATHS)
@pytest.mark.parametrize("equal_rewards", [False, True])
def test_every_grpo_loss_backward_and_zero_advantage(model, tokenizer, script, equal_rewards, monkeypatch):
    module = load_script(script)
    loss_fn = next(value for name, value in vars(module).items()
                   if name.startswith("compute_grpo_loss") and callable(value))
    counter = iter(range(20))
    def rollout(**kwargs):
        n = next(counter)
        ids = torch.tensor([8, 9, 10, 11 + n % 2, 12])
        correct = equal_rewards or n % 2 == 0
        text = r"\boxed{2}" if correct else r"\boxed{3}"
        result = (ids, 3, text)
        return (*result, []) if "deepseek" in script else result
    if hasattr(module, "sample_responses_batched"):
        monkeypatch.setattr(module, "sample_responses_batched",
                            lambda **kw: [rollout() for _ in range(kw["batch_size"])])
    else:
        monkeypatch.setattr(module, "sample_response", rollout)
    parameters = inspect.signature(loss_fn).parameters
    kwargs = dict(model=model, tokenizer=tokenizer, example={"problem": "1+1", "answer": "2"},
                  device="cpu", num_rollouts=2, max_new_tokens=2)
    if "old_model" in parameters:
        kwargs["old_model"] = copy.deepcopy(model)
    if "ref_model" in parameters:
        kwargs["ref_model"] = copy.deepcopy(model)
    if "skip_zero_adv" in parameters:
        kwargs["skip_zero_adv"] = equal_rewards
    result = loss_fn(**kwargs)
    assert all(torch.isfinite(torch.tensor(result[key])).all()
               for key in ("loss", "advantages"))
    assert len(result["samples"]) == 2
    if result["loss_tensor"] is not None:
        result["loss_tensor"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not equal_rewards:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    with pytest.raises(ValueError, match="at least two"):
        loss_fn(**{**kwargs, "num_rollouts": 1})


@pytest.mark.parametrize("script", GRPO_PATHS[1:])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_every_rollout_sampler_uses_native_cache(model, tokenizer, script, temperature):
    module = load_script(script)
    kwargs = dict(model=model, tokenizer=tokenizer, prompt="answer", device="cpu",
                  max_new_tokens=2, temperature=temperature, top_p=0.9)
    if hasattr(module, "sample_responses_batched"):
        outputs = module.sample_responses_batched(**kwargs, batch_size=2)
    else:
        if "deepseek" in script:
            kwargs["top_k"] = 3
        outputs = [module.sample_response(**kwargs)]
    for output in outputs:
        ids, prompt_len, _ = output[:3]
        assert ids.numel() - prompt_len == 2
        if temperature == 0:
            expected = ch02.generate_text_basic_cache(model, ids[:prompt_len].unsqueeze(0), 2)
            torch.testing.assert_close(ids[prompt_len:], expected[0])
        if "deepseek" in script:
            logp, entropy = module.sequence_logprob_and_entropy(
                model, ids, prompt_len, keep_masks=output[3], return_entropy=True
            )
            assert torch.isfinite(logp) and torch.isfinite(entropy)


def test_single_and_batched_distillation_loss_gradients_agree(model, tmp_path):
    batched = load_script("ch08/04_train_with_distillation/distill_batched.py")
    single = load_script("ch08/04_train_with_distillation/distill.py")
    examples = [{"token_ids": [2, 3, 4, 5, 6], "prompt_len": 2},
                {"token_ids": [7, 8, 9], "prompt_len": 1}]
    expected = torch.stack([ch08.compute_example_loss(model, e, "cpu") for e in examples]).mean()
    expected.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    actual, count = batched.compute_batch_loss(model, examples, pad_id=0, device="cpu")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[n], atol=1e-5, rtol=1e-4)
    assert count == 5
    for module in (ch08, single):
        torch.testing.assert_close(module.compute_example_loss(model, examples[0], "cpu"),
                                   ch08.compute_example_loss(model, examples[0], "cpu"))
        with pytest.raises(ValueError, match="at least one"):
            module.evaluate_examples(model, [], "cpu")
    assert batched.evaluate_examples_batched(model, examples, 2, 0, "cpu") == pytest.approx(actual.item())


@pytest.mark.parametrize("script", GRPO_PATHS)
def test_every_grpo_training_loop_saves_reloadable_checkpoint(model, tokenizer, script, tmp_path, monkeypatch):
    model.cfg["context_length"] = 256  # Rendered instructions plus chat tokens
    module = load_script(script)
    for name in ("LOG_PATH", "METRICS_LOG_PATH", "CSV_LOG_PATH"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, tmp_path / f"{name}.txt")
    parameters = inspect.signature(module.train_rlvr_grpo).parameters
    kwargs = dict(model=model, tokenizer=tokenizer, math_data=[{"problem": "1+1", "answer": "2"}],
                  device="cpu", steps=1, num_rollouts=2, max_new_tokens=2,
                  checkpoint_every=1, checkpoint_dir=tmp_path)
    if "math500_eval_data" in parameters:
        kwargs["math500_eval_data"] = []
    if "ref_model" in parameters:
        kwargs["ref_model"] = copy.deepcopy(model)
    if "inner_epochs" in parameters:
        kwargs["inner_epochs"] = 1
    if "active_sampling" in parameters:
        kwargs["active_sampling"] = False
    if "csv_log_path" in parameters:
        kwargs["csv_log_path"] = tmp_path / "metrics.csv"
    # The FSDP module's helper is exercised unsharded; real collectives need CUDA.
    if "fsdp" in script:
        def save_unsharded(model, checkpoint_dir, step, is_main=True, suffix=""):
            return ch06.save_checkpoint(model, checkpoint_dir, step, suffix)
        monkeypatch.setattr(module, "save_checkpoint", save_unsharded)
        monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
    module.train_rlvr_grpo(**kwargs)
    paths = list(tmp_path.glob("*.pth"))
    assert paths
    restored = load_pretrained_model(paths[0], cfg=model.cfg)
    ids = torch.tensor([[2, 3]])
    torch.testing.assert_close(restored(ids), model.eval()(ids))
