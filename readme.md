统一生成式检索 (Generative Retrieval, GR) 与点击率预测 (CTR) 的多任务学习框架。其核心思想是利用共享的序列表达能力，同时驱动两种不同性质的推荐任务。


```Text
UniGCR_Repo/
├── ds_config.json          # DeepSpeed 配置文件
├── requirements.txt        # 依赖列表 (含安装顺序说明)
├── run.py                  # 启动入口
└── src/
    ├── __init__.py
    ├── config.py           # 全局配置 (Dataclass)
    ├── data.py             # UniversalDataset & DataLoader
    ├── grid_utils.py       # Semantic ID 映射与反查工具
    ├── model.py            # 模型核心 (InputLayer, HSTU, Heads)
    ├── trainer.py          # 训练循环, EarlyStop, Eval
    └── utils.py            # Metrics, Distributed Utils


配置
PyTorch 带CUDA
# 示例：安装 PyTorch 2.1 + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
安装Flash Attention 2
pip install packaging ninja
pip install flash-attn --no-build-isolation

安装其他依赖包
# 安装 DeepSpeed, Scikit-learn 等
pip install deepspeed numpy pandas scikit-learn tqdm wget triton

# 安装 Meta 的 generative-recommenders (HSTU)
pip install git+https://github.com/facebookresearch/generative-recommenders.git@main


必要的准备工作 (Checklist)
在运行之前，请确认以下文件存在：
GRID Mapping File: data/beauty/semantic_ids.json。这是由 GRID 预处理生成的，格式应为 { "item_id": [code1, code2, code3], ... }。
Config Adjustments: 在 run.py 中，请务必修改 conf.num_atomic_items 为你数据集真实的 Item 总数，否则 Embedding 层会报错或越界。
# 指定可见设备
# 即使是单卡，也建议用 torchrun 启动以保持环境一致
torchrun --nproc_per_node=1 run.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --grid_mapping data/beauty/semantic_ids.json

评价指标完善：
GR: 保持了 HitRate 和 NDCG。
CTR: 新增了 AUC 和 LogLoss。通过 gather_tensors 确保了在多 GPU 环境下，AUC 是基于全局数据计算的，而不是局部 AUC 的平均值（那是不准确的）。

```

UniGCR × GRID × HSTU（GR-Only）对接改动说明（Amazon Beauty / Sports / Toys）

```Text
1) 数据（来自 GRID）

- 使用的数据集：Amazon Beauty（后续可切换 sports / toys）

- Item 到 Semantic ID 的映射（GRID Step2/3 输出）
  - 路径：data/GRID/semantic_ids/beauty/part-00000.pkl
  - item 数量：12101
  - 参数设置：num_hierarchies = 3，codebook_width = 256
  - 说明：每个 item 被映射为 3 个层级的 semantic token

- User item sequences
  - 训练集：data/GRID/outputs/beauty_user_item_sequences/beauty_user_item_sequences_training.pt
  - 验证集：data/GRID/outputs/beauty_user_item_sequences/beauty_user_item_sequences_evaluation.pt
  - 测试集：data/GRID/outputs/beauty_user_item_sequences/beauty_user_item_sequences_testing.pt
  - user sequences 总数：22363


2) HSTU 依赖与兼容处理

- 已成功安装 generative-recommenders，但存在 import 兼容问题
  - 解决方式：将 generative-recommenders/generative_recommenders/research 目录复制到本项目中
  - 并重命名为 research_hstu
  - 使用方式：从 research_hstu.modeling.sequential.hstu 中导入 HSTU
  - 说明：该库中不存在 HSTUConfig 类，因此采用显式参数初始化 HSTU

- 去除 FBGEMM 依赖
  - 原因：fbgemm 安装失败
  - 解决方式：在本地实现替代函数以避免依赖 fbgemm
  - 替代实现包括：
    - safe_complete_cumsum
    - safe_jagged_to_padded_dense
    - safe_dense_to_jagged


3) Data pipeline 改动（对齐 HSTU 的右 padding）

- padding 方式从左 padding 改为右 padding，以与 HSTU 和 GR 的训练行为保持一致

- 训练样本中构造的主要字段包括：
  - sem_history：full_seq 去掉最后一个 token，长度为 L-1
  - sem_target：full_seq 向右偏移一位，长度为 L-1，用于 next-token 预测
  - sem_target_eval：目标 item 对应的 semantic token 序列，用于评估和 beam search
  - ctr_pos_codes：CTR 相关字段保留，但在当前 GR-only 设置中不使用


4) Model forward 的关键逻辑（Embedding 视角）

- 输入张量
  - sem_history 的形状为 (B, L-1)，表示 token ids

- 在 forward 中
  - 将 sem_history 在末尾补一个 dummy padding token，构造长度为 L 的输入序列
  - embedding 后送入 HSTU
  - HSTU 输出 u_seq，形状为 (B, L, D)

- 训练时使用的 user 表示与 logits
  - 使用 u_seq 去掉最后一个时间步得到 u_history，形状为 (B, L-1, D)
  - 将 u_history 输入到 GR head，得到 logits，形状为 (B, L-1, V)
  - 使用 sem_target 计算 token-level cross entropy loss，即 sequence-level next-token 训练

- 额外提取的表示
  - 从 u_seq 中提取最后一个真实 token 对应的 hidden state，得到 u_last，形状为 (B, D)
  - u_last 主要用于 beam search（生成候选 semantic id 或 hard negatives），以及未来 CTR 扩展

- Attention 机制
  - HSTU 使用 causal attention mask
  - 确保每个位置只能访问历史信息，不会看到未来或 target 信息


5) 已验证的踩坑记录

- 曾尝试仅使用 u_last（最后一个 token 的 embedding）直接计算 logits 和 loss
  - 观察到 loss 很快下降到接近 0
  - 但 HitRate 和 NDCG 等指标几乎始终为 0
- 结论
  - 该方式会导致训练信号与目标错位或产生信息泄漏
  - 已回退到基于 u_seq 去掉最后一个时间步的 sequence-level next-token 训练方式


6) 运行方式（GR-only）

- 当前仅支持 GR，不包含 CTR
- 运行设置为单卡、单进程，不使用分布式训练

- 主入口脚本
  - run_gr_only.py

- 主要修改文件
  - src/data.py：读取 GRID 的 pt 文件，使用右 padding，输出 sem_history 和 sem_target
  - src/model.py：接入 HSTU，实现 GR-only forward 和 beam search
  - src/trainer.py：关闭 CTR loss，仅训练 GR
  - src/config.py：仅调整 hstu_layers 和 hstu_heads 等必要参数
```
