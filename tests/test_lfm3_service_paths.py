"""Offline contracts for optional datasets, teacher APIs and chat UI handlers."""

import asyncio
import io
import json
import sys
import types
import subprocess

import pytest
import torch

from reasoning_from_scratch import ch02, ch03, appendix_f
from test_lfm3_chapter_paths import load_script


def test_mmlu_continuations_do_not_retokenize_or_rewrap_prompt(monkeypatch):
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *a, **kw: []
    datasets.get_dataset_config_names = lambda *a: ["fixture"]
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    class Tokenizer:
        def encode(self, text, chat_wrapped=None):
            assert chat_wrapped is False
            return [ord(text[0]) - ord("A") + 4] + ([8] if len(text) > 1 else [])
    class Model:
        def __call__(self, ids):
            logits = torch.zeros(*ids.shape, 12)
            logits[..., 6] = 4  # Choice C
            return logits
    prompt = torch.tensor([[1, 2]])
    example = {"choices": ["one", "two", "three", "four"]}
    for path, name in [("2_logprob.py", "predict_choice_logprobs"),
                       ("3_teacher_forcing.py", "predict_choice_teacher_forced")]:
        module = load_script("chF/02_mmlu/" + path)
        args = [Model(), Tokenizer(), prompt, "prompt", [1, 2], example]
        pred, scores = getattr(module, name)(*args)
        assert pred == "C" and all(torch.isfinite(torch.tensor(v)) for v in scores.values())


@pytest.mark.parametrize("text, expected", [("Answer: B", "B"), ("capital", None),
                                            ("C. explanation", "C"), ("D", "D")])
def test_letter_matching_ignores_letters_inside_words(monkeypatch, text, expected):
    monkeypatch.setattr(appendix_f, "generate_text_basic_stream_cache",
                        lambda **kw: iter([torch.tensor([[1]])]))
    tokenizer = types.SimpleNamespace(eos_token_id=0, decode=lambda ids: text)
    assert appendix_f.predict_choice(None, tokenizer, torch.tensor([[1]])) == expected


@pytest.mark.parametrize("provider", ["ollama", "openrouter"])
def test_teacher_request_retry_response_and_resume(provider, monkeypatch, tmp_path):
    module = load_script(f"ch08/02_generate_distillation_data/generate_with_{provider}.py")
    body = ({"message": {"content": r"\boxed{2}", "thinking": "1+1=2"}}
            if provider == "ollama" else
            {"choices": [{"message": {"content": r"\boxed{2}", "reasoning": "1+1=2"}}]})
    requests = []
    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            raise module.error.URLError("fixture transient failure")
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr(module.request, "urlopen", urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    kwargs = dict(prompt="What is 1+1?", model="fixture-teacher", max_new_tokens=8,
                  temperature=0.2, timeout=1, max_retries=2, retry_delay=0)
    if provider == "ollama":
        kwargs["url"] = "http://fixture.invalid"
    else:
        kwargs.update(api_key="fixture-not-a-real-key", top_p=0.9)
    result = getattr(module, f"query_{provider}_chat")(**kwargs)
    assert result["message_content"] == r"\boxed{2}"
    assert len(requests) == 2
    out = tmp_path / "rows.json"
    module.write_rows_json_incremental([result], out)
    assert module.load_resume_rows(out) == [result]


@pytest.mark.parametrize("multiturn", [False, True])
@pytest.mark.parametrize("stop_immediately", [False, True])
def test_chat_handlers_encode_and_generate_with_native_lfm3(multiturn, stop_immediately, monkeypatch, tmp_path):
    from test_lfm3 import tiny_cfg as cfg_fixture, tokenizer as tok_fixture
    from reasoning_from_scratch.lfm3 import LFM3Model
    cfg = cfg_fixture.__wrapped__()
    cfg["context_length"] = 128
    model = LFM3Model(cfg).eval()
    tokenizer = tok_fixture.__wrapped__(tmp_path)
    # Control the emitted word/EOS without skipping the hybrid forward pass.
    # UI tests must not depend on random model quality or test ordering.
    def controlled_tokens(_, args, output):
        output.fill_(-1000)
        output[..., tokenizer.eos_token_id if stop_immediately else 8] = 1000
        return output
    model.register_forward_hook(controlled_tokens)
    fake = types.ModuleType("chainlit")
    session = {}
    fake.user_session = types.SimpleNamespace(get=session.get, set=session.__setitem__)
    fake.on_chat_start = fake.on_message = lambda fn: fn
    messages = []
    class Message:
        def __init__(self, content):
            self.content = content
            messages.append(self)
        async def send(self):
            pass
        async def stream_token(self, piece):
            self.content += piece
        async def update(self):
            pass
    fake.Message = Message
    monkeypatch.setitem(sys.modules, "chainlit", fake)
    monkeypatch.setattr(ch02, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ch03, "load_model_and_tokenizer", lambda **kw: (model, tokenizer))
    suffix = "_multiturn" if multiturn else ""
    module = load_script(f"chG/01_main-chapter-code/qwen3_chat_interface{suffix}.py")
    module.MAX_NEW_TOKENS = 2
    if multiturn:
        asyncio.run(module.on_start())
    asyncio.run(module.main(Message("answer")))
    assert messages[-1].content == ("" if stop_immediately else "answeranswer")
    if multiturn:
        assert [m["role"] for m in session["history"]] == ["system", "user", "assistant"]
        asyncio.run(module.main(Message("answer again")))
        assert len(session["history"]) == 5


@pytest.mark.parametrize("name", ["1_elo_leaderboard.py", "2_bradley_terry_leaderboard.py"])
def test_leaderboard_cli_validates_input_and_ranks_votes(name, tmp_path):
    from test_lfm3_chapter_paths import ROOT

    path = ROOT / "chF/03_leaderboards" / name
    missing = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert missing.returncode == 2 and "--path" in missing.stderr
    votes = tmp_path / "votes.json"
    votes.write_text(json.dumps([["A", "B"], ["A", "C"], ["B", "C"]]))
    valid = subprocess.run([sys.executable, str(path), "--path", str(votes)],
                           capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    ranks = [line.strip().split()[1] for line in valid.stdout.splitlines()
             if line.strip().startswith(("1.", "2.", "3."))]
    assert ranks == ["A", "B", "C"]


def test_bradley_terry_handles_empty_votes():
    module = load_script("chF/03_leaderboards/2_bradley_terry_leaderboard.py")
    assert module.bradley_terry_torch([], "cpu") == {}
