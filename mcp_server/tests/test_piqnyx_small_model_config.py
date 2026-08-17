from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config.schema import LLMConfig


def test_small_model_defaults_to_the_main_model():
    # Unset means "same model", which is the behaviour every existing deployment
    # already has: exposing the key must not change anything by itself.
    config = LLMConfig(model='deepseek-v4-flash')
    assert config.small_model is None
    assert (config.small_model or config.model) == 'deepseek-v4-flash'


def test_small_model_can_be_pointed_at_a_cheaper_model():
    # The mechanical calls — entity attributes, edge timestamps, edge dedup — run
    # on every batch, so a reasoning model there is paid for on every batch.
    config = LLMConfig(model='deepseek-v4-flash', small_model='qwen3-4b')
    assert (config.small_model or config.model) == 'qwen3-4b'
    assert config.model == 'deepseek-v4-flash'
