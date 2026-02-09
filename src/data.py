import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from .grid_utils import GridMapper
import os
import torch.distributed as dist

class UniversalDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data = []

        self.grid_mapper = None
        if config.use_semantic_seq:
            self.grid_mapper = GridMapper(
                config.grid_mapping_path,
                config.sem_id_layers,
                config.sem_id_codebook_size
            )
            self.config.sem_total_vocab = self.grid_mapper.total_vocab_size

        self._load_data(config.data_path)

    def _load_data(self, path):
        """
        加载数据入口。支持自动识别 GRID (.pt) 格式。
        """
        filename = [f for f in os.listdir(path)  if self.mode in f][0]
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] Data path {filepath} not found. Falling back to Sim Mode...")
            self._generate_sim_data()
            return

        print(f"Loading data from {filepath}...")

        # 1. 识别并处理 GRID 格式 (.pt 结尾且内容为 Dict)
        if filepath.endswith('.pt'):
            try:
                user2seq = torch.load(filepath, map_location="cpu")
                if isinstance(user2seq, dict):
                    print(f"Detected GRID sequence format. Converting...")
                    self._convert_grid_to_internal(user2seq)
                    return
            except Exception as e:
                print(f"Error loading .pt file: {e}")

        # 2. 如果是其他格式（如 JSON/CSV），可以在此处扩展
        # else if path.endswith('.json'): ...

        # 如果格式未知，回退到模拟
        print(f"Unknown data format or loading failed. Using Sim Mode.")
        self._generate_sim_data()

    def _convert_grid_to_internal(self, user2seq):
        """
        将 GRID 格式 {uid: [item1, item2, ...]} 转换为内部统一的样本格式。
        """
        for uid, seq in user2seq.items():
            if len(seq) < 2:
                continue

            # 构造内部统一的样本格式，将最后一位作为 Target
            sample = {
                'user_id': uid,
                'target_item': seq[-1],  # 取最后一个 item 作为目标
                'sem_seq': seq[:-1]  # 前面的作为历史序列
            }

            # 如果开启了 Atomic 序列特征，同步使用原始 ID 序列
            if self.config.use_atomic_seq:
                sample['atom_seq'] = seq[:-1]

            # 暂时不包含 Profile 特征，GRID 默认只提供序列
            self.data.append(sample)
        print(f"Successfully converted {len(self.data)} GRID samples.")

    def _generate_sim_data(self):
        """
        模拟数据生成逻辑，用于调试或无数据时运行。
        """
        print(f"Generating Simulation Data (1000 samples)...")
        for _ in range(1000):
            sample = {}
            target_item = np.random.randint(1, 1000)
            sample['target_item'] = target_item

            if self.config.use_semantic_seq:
                sample['sem_seq'] = np.random.randint(1, 1000, 20).tolist()

            if self.config.use_atomic_seq:
                sample['atom_seq'] = np.random.randint(1, self.config.num_atomic_items + 1, 20).tolist()

            if self.config.use_cat_profile:
                sample['cat_feats'] = [np.random.randint(0, v) for v in self.config.cat_feature_vocab_sizes]
            if self.config.use_num_profile:
                sample['num_feats'] = np.random.rand(self.config.num_feature_size).tolist()

            self.data.append(sample)

    def __getitem__(self, idx):
        raw_item = self.data[idx]
        output = {}

        # 目标 Item 的 Semantic Codes (用于 GR Target 和 CTR Positive)
        tgt_codes = self.grid_mapper.get_codes(raw_item['target_item'])

        # ===========================
        # 1. Semantic Sequence & Target
        # ===========================
        if self.config.use_semantic_seq:
            seq_codes = self.grid_mapper.flatten_sequence(raw_item['sem_seq'])
            full_seq = seq_codes + tgt_codes

            max_len = self.config.max_seq_len
            if len(full_seq) > max_len:
                full_seq = full_seq[-max_len:]
            else:
                full_seq = full_seq + [0] * (max_len - len(full_seq))

            # 模型输入: tokens [0, L-1]
            output['sem_history'] = torch.tensor(full_seq[:-1], dtype=torch.long)
            # 训练标签 (预测下一个 token): tokens [1, L]
            output['sem_target'] = torch.tensor(full_seq[1:], dtype=torch.long)

            # === 新增：用于 Eval 的纯净 history（不包含 target codes） ===
            # 只包含历史 item 的 codes，用于 beam search 生成
            pure_history = seq_codes  # 不含 target
            if len(pure_history) > max_len - self.config.sem_id_layers:
                pure_history = pure_history[-(max_len - self.config.sem_id_layers):]
            else:
                pure_history = pure_history + [0] * (max_len - self.config.sem_id_layers - len(pure_history))
            output['sem_history_eval'] = torch.tensor(pure_history, dtype=torch.long)

            # 用于 CTR 任务的正样本表示
            output['ctr_pos_codes'] = torch.tensor(tgt_codes, dtype=torch.long)
            output['sem_target_eval'] = torch.tensor(tgt_codes, dtype=torch.long)
            output['target_item'] = raw_item['target_item']

        # ===========================
        # 2. Atomic Sequence (仅作上下文特征)
        # ===========================
        if self.config.use_atomic_seq and 'atom_seq' in raw_item:
            seq = raw_item['atom_seq']
            max_len = self.config.max_atomic_len
            if len(seq) > max_len:
                seq = seq[-max_len:]
            else:
                seq = [0] * (max_len - len(seq)) + seq
            output['atom_history'] = torch.tensor(seq, dtype=torch.long)

        # ===========================
        # 3. User Profiles
        # ===========================
        if self.config.use_cat_profile and 'cat_feats' in raw_item:
            output['cat_feats'] = torch.tensor(raw_item['cat_feats'], dtype=torch.long)
        if self.config.use_num_profile and 'num_feats' in raw_item:
            output['num_feats'] = torch.tensor(raw_item['num_feats'], dtype=torch.float)

        return output

    def __len__(self):
        return len(self.data)


def get_dataloaders(config):
    train_ds = UniversalDataset(config, mode='train')
    val_ds = UniversalDataset(config, mode='eva')

    # 同步动态生成的词表大小
    if config.use_semantic_seq:
        val_ds.config.sem_total_vocab = train_ds.config.sem_total_vocab

    if dist.is_available() and dist.is_initialized():
        # 分布式模式
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        # 单卡模式
        train_sampler = None
        val_sampler = None

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, sampler=val_sampler, num_workers=2)

    return train_dl, val_dl