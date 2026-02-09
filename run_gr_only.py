import sys
import os
import argparse
import torch
import torch.nn as nn
from src.config import UniGCRConfig
from src.data import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed, setup_distributed, is_main_process
import deepspeed

def parse_args():
    parser = argparse.ArgumentParser(description="Uni-GCR GR-Only Training")
    parser.add_argument('--local_rank', type=int, default=-1)
    # parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
    parser.add_argument('--data_path', type=str, default='data/GRID/outputs/beauty_user_item_sequences')
    parser.add_argument('--grid_mapping', type=str, default='data/GRID/semantic_ids/beauty/part-00000.pkl')
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 初始化分布式环境
    # setup_distributed()

    # 2. 初始化配置
    conf = UniGCRConfig()

    # --- [GR-ONLY 核心配置] ---
    conf.enable_ctr = False  # 强制关闭 CTR 任务
    conf.use_semantic_seq = True  # 开启 GRID 语义预测
    conf.use_atomic_seq = False  # 暂时不使用 Atomic ID 简化任务
    conf.use_cat_profile = False  # 暂时不使用画像简化任务
    conf.use_num_profile = False

    # 动态参数同步
    conf.grid_mapping_path = args.grid_mapping
    conf.data_path = args.data_path

    set_seed(conf.seed)

    # 3. 准备数据 (这里会触发 GridMapper 加载)
    if is_main_process():
        print(f"Loading Data from {conf.data_path}...")
        print(f"Using GRID Mapping: {conf.grid_mapping_path}")

    train_dl, val_dl = get_dataloaders(conf)

    if is_main_process():
        print(f"GRID Vocabulary Size: {conf.sem_total_vocab}")

    # 5. 初始化模型 (UniGCRModel 内部会自动加载补丁后的 HSTU)
    model = UniGCRModel(conf)

    # 6. 使用 Trainer 运行
    # 由于 conf.enable_ctr = False，Trainer 内部的 loss 计算会自动跳过 CTR 部分
    trainer = UniGCRTrainer(
        config=conf,
        args=args,
        model=model,
        train_loader=train_dl,
        val_loader=val_dl
    )

    if is_main_process():
        print("Model Initialized for GR-Only. Starting Training...")

    # 7. 开始训练
    trainer.train()


if __name__ == "__main__":
    main()