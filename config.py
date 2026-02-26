from dataclasses import dataclass, field
from typing import Literal, Optional

GPT2_VOCAB_SIZE = 50257


@dataclass
class ModelConfig:
    name: str = "tiny-reasoner"
    vocab_size: int = GPT2_VOCAB_SIZE
    n_layer: int = 12
    n_embd: int = 512
    ctx_len: int = 2048


@dataclass
class RWKVConfig(ModelConfig):
    architecture: Literal["rwkv7"] = "rwkv7"
    head_size: int = 64

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_size


@dataclass
class TransformerConfig(ModelConfig):
    architecture: Literal["transformer"] = "transformer"
    n_head: int = 8
    n_intermediate: int = 1365
    n_query_groups: int = 4
    rotary_pct: float = 0.25

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


PRESETS = {
    "micro": TransformerConfig(
        name="micro",
        n_layer=6,
        n_embd=384,
        n_head=6,
        n_intermediate=1024,
        n_query_groups=2,
    ),
    "tiny": TransformerConfig(
        name="tiny",
        n_layer=8,
        n_embd=512,
        n_head=8,
        n_intermediate=1365,
        n_query_groups=4,
    ),
    "small": TransformerConfig(
        name="small",
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_intermediate=2048,
        n_query_groups=4,
    ),
    "tiny-rwkv": RWKVConfig(name="tiny-rwkv", n_layer=12, n_embd=512, head_size=64),
    "small-rwkv": RWKVConfig(name="small-rwkv", n_layer=18, n_embd=768, head_size=64),
}


def get_config(name: str):
    if name in PRESETS:
        return PRESETS[name]
    raise ValueError(f"Unknown config: {name}. Available: {list(PRESETS.keys())}")
