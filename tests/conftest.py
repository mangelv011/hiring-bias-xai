import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    PreTrainedTokenizerFast,
)

from hiring_bias_xai import data

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
SPECIAL = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]


def _tokenizer(max_length):
    vocab = {token: i for i, token in enumerate(SPECIAL + WORDS)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        model_max_length=max_length,
    )


@pytest.fixture
def tiny_network():
    """Return make(): a random DeBERTa-v2 with 2 layers and 28 labels, built offline."""

    def make(seed=0):
        torch.manual_seed(seed)
        config = DebertaV2Config(
            vocab_size=len(SPECIAL + WORDS),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
            pad_token_id=0,
            num_labels=len(data.OCCUPATIONS),
        )
        return DebertaV2ForSequenceClassification(config)

    return make


@pytest.fixture
def tiny_tokenizer():
    return _tokenizer
