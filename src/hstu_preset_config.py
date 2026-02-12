# hstu_presets.py
from dataclasses import dataclass

@dataclass
class HSTUPreset:
    embed_dim: int
    hstu_layers: int
    hstu_heads: int
    dropout: float
    max_seq_len: int
    train_batch_size: int
    eval_batch_size: int
    lr: float
    epochs: int
    head_dim: int
    qk_dim: int

# follow research_hstu/configs/hstu-sampled-softmax-n512-final.gin
HSTU_BOOKS_N512 = HSTUPreset(
    embed_dim=64,
    hstu_layers=4,
    hstu_heads=4,
    dropout=0.5,
    max_seq_len=50*3, #考虑到我们使用3层SID
    train_batch_size=128,
    eval_batch_size = 128,
    lr = 1e-3,
    epochs = 200,
    head_dim = 16,  # 对应 dv
    qk_dim = 16,  # 对应 dqk
)

# follow research_hstu/configs/hstu-sampled-softmax-n512-large-final.gin
HSTU_BOOKS_N512_LARGE = HSTUPreset(
    embed_dim=64,
    hstu_layers=16,
    hstu_heads=8,
    dropout=0.5,
    max_seq_len=50*3, #考虑到我们使用3层SID
    train_batch_size=128,
    eval_batch_size=128,
    lr=1e-3,
    epochs=200,
    head_dim = 8,  # 对应 dv
    qk_dim = 8,  # 对应 dqk
)
