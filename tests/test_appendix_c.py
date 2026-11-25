"""The appendix teaches the same LFM3 implementation used in the chapters."""

from pathlib import Path

import torch

from conftest import import_definitions_from_notebook
from reasoning_from_scratch.ch02 import generate_text_basic, generate_text_basic_cache
from reasoning_from_scratch.lfm3 import LFM3_CONFIG_1_7B


def test_appendix_generation_with_and_without_cache():
    root = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(root / "chC/01_main-chapter-code/chC_main.ipynb", "chC_defs")
    cfg = dict(LFM3_CONFIG_1_7B, vocab_size=32, emb_dim=16, hidden_dim=32,
               n_heads=2, n_kv_groups=1, head_dim=8, n_layers=3,
               layer_types=("conv", "sliding_attention", "full_attention"),
               dtype=torch.float32, sliding_window=4)
    model = mod.LFM3Model(cfg)
    ids = torch.tensor([[1, 2, 3]])
    plain = generate_text_basic(model, ids, max_new_tokens=6)
    cached = generate_text_basic_cache(model, ids, max_new_tokens=6)
    torch.testing.assert_close(plain, cached)
