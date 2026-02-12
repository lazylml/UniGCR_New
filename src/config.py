from dataclasses import dataclass, field
from typing import List
import torch
from .hstu_preset_config import HSTUPreset

@dataclass
class UniGCRConfig:
    # --- [任务开关] ---
    enable_ctr: bool = True           # 是否开启 CTR 联合训练
    use_semantic_seq: bool = True     # 是否使用 GRID Semantic ID
    use_atomic_seq: bool = False      # 是否使用 Atomic ID
    use_cat_profile: bool = True      # 是否使用类别用户画像
    use_num_profile: bool = True      # 是否使用数值用户画像
    
    # --- [CTR 模块微调] ---
    ctr_use_self_attn: bool = True    # Candidate-Aware Self-Attention
    ctr_use_cross_attn: bool = True   # User-Centric Cross-Attention
    
    # --- [Semantic ID / GRID] ---
    sem_id_layers: int = 3
    sem_id_codebook_size: int = 256
    grid_mapping_path: str = "data/beauty/semantic_ids.json"

    # --- [Atomic ID] ---
    num_atomic_items: int = 0
    max_atomic_len: int = 50
    
    # --- [User Profile] ---
    cat_feature_vocab_sizes: List[int] = field(default_factory=lambda: [1000, 20]) 
    num_feature_size: int = 5

    # --- [数据参数] ---
    num_workers: int = 8

    # --- [模型参数] ---
    embed_dim: int = 64
    max_seq_len: int = 150  # 需足够容纳 (Sem_Len + Atom_Len + Profile_Len)
    hstu_layers: int = 4 # default 2
    hstu_heads: int = 4 # default 2
    dropout: float = 0.5
    attn_alpha: float = 1.7
    head_dim: int = 16  # 对应 dv
    qk_dim: int = 16  # 对应 dqk

    # --- [训练参数] ---
    patience: int = 3
    train_batch_size: int = 128 #default 64
    lr: float = 1e-3
    epochs: int = 200 #50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ===== Eval strategy =====
    eval_batch_size: int = 128 #default 64
    eval_interval: int = 50        # 每 N 个 epoch 才做 ranking eval
    eval_with_ranking: bool = True

    # --- [Uni-GCR Loss] ---
    memory_bank_size: int = 20
    num_hard_negatives: int = 5
    temp: float = 0.07  
    loss_alpha: float = 1.0 
    loss_beta: float = 0.5 
    
    # --- [运行时动态填充] ---
    sem_total_vocab: int = 0


def apply_hstu_preset(conf, preset: HSTUPreset):
    conf.embed_dim = preset.embed_dim
    conf.hstu_layers = preset.hstu_layers
    conf.hstu_heads = preset.hstu_heads
    conf.dropout = preset.dropout
    conf.max_seq_len = preset.max_seq_len
    conf.train_batch_size = preset.train_batch_size
    conf.eval_batch_size = preset.eval_batch_size
    conf.lr = preset.lr
    conf.epochs = preset.epochs
    conf.head_dim = preset.head_dim
    conf.qk_dim = preset.qk_dim
