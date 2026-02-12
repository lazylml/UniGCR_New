import sys
import os
import argparse
import torch
import torch.nn as nn
from src.config import UniGCRConfig, apply_preset_by_name
from src.data import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed, setup_distributed, is_main_process
# import deepspeed
import wandb
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Uni-GCR GR-Only Training")
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def main():
    # ===== TF32 加速（A6000 强烈推荐）=====
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 可选：benchmark（对固定 shape 有帮助）
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    # 1. 初始化分布式环境
    # setup_distributed()

    # 2. 初始化配置
    conf = UniGCRConfig()
    if conf.use_hstu_config:
        apply_preset_by_name(conf)

    # --- [GR-ONLY 核心配置] ---
    conf.enable_ctr = False  # 强制关闭 CTR 任务
    conf.use_semantic_seq = True  # 开启 GRID 语义预测
    conf.use_atomic_seq = False  # 暂时不使用 Atomic ID 简化任务
    conf.use_cat_profile = False  # 暂时不使用画像简化任务
    conf.use_num_profile = False

    # 动态参数同步
    # conf.grid_mapping_path = args.grid_mapping
    # conf.data_path = args.data_path

    set_seed(conf.seed)

    # 3. 准备数据 (这里会触发 GridMapper 加载)
    if is_main_process():
        print(f"Loading Data from {conf.data_path}...")
        print(f"Using GRID Mapping: {conf.grid_mapping_path}")

    train_dl, val_dl, test_dl = get_dataloaders(conf)

    if is_main_process():
        print(f"GRID Vocabulary Size: {conf.sem_total_vocab}")


    # 5. 初始化模型 (UniGCRModel 内部会自动加载补丁后的 HSTU)
    model = UniGCRModel(conf)

    if is_main_process():
        timestamp = datetime.now().strftime("%m%d-%H%M")
        run_name = (
            f"GR_"
            f"L{conf.hstu_layers}"
            f"H{conf.hstu_heads}"
            f"D{conf.embed_dim}"
            f"Dh{conf.embed_dim // conf.hstu_heads}"
            f"DP{str(conf.dropout).replace('.', '')}"
            f"_{timestamp}"
        )
        wandb.init(
            entity="unigcr",
            project="unigcr",
            name=run_name,
            config=vars(conf),
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # 6. 使用 Trainer 运行
    # 由于 conf.enable_ctr = False，Trainer 内部的 loss 计算会自动跳过 CTR 部分
    trainer = UniGCRTrainer(
        config=conf,
        args=args,
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        test_loader=test_dl
    )

    if is_main_process():
        print("Model Initialized for GR-Only. Starting Training...")

    # 7. 开始训练
    trainer.train()

    print("Loading best model for final test...")
    trainer.load_best()

    eval_metrics = trainer.evaluate(topk=[5, 10], with_ranking=True, mode='test')
    if is_main_process():
        log_str = "Eval: "
        log_str += f"GR_Loss={eval_metrics.get('test_gr_loss', 0):.4f} "

        if 'Hit@5' in eval_metrics:
            log_str += f"Hit@5={eval_metrics['Hit@5']:.4f} "
        if 'Hit@10' in eval_metrics:
            log_str += f"Hit@10={eval_metrics['Hit@10']:.4f} "
        if 'NDCG@5' in eval_metrics:
            log_str += f"NDCG@5={eval_metrics['NDCG@5']:.4f} "
        if 'NDCG@10' in eval_metrics:
            log_str += f"NDCG@10={eval_metrics['NDCG@10']:.4f} "

        if conf.enable_ctr:
            log_str += f"AUC={eval_metrics.get('AUC', 0):.4f} LogLoss={eval_metrics.get('LogLoss', 0):.4f}"

        print(log_str)

    if is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()