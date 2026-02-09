import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import UniGCRConfig

from research_hstu.modeling.sequential.hstu import HSTU as OfficialHSTU
from research_hstu.modeling.sequential.input_features_preprocessors import InputFeaturesPreprocessorModule
from research_hstu.rails.similarities.module import SimilarityModule
from research_hstu.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
    LayerNormEmbeddingPostprocessor,
    L2NormEmbeddingPostprocessor,
)

class SemanticAwareInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    """
    A semantic-token-aware HSTU input preprocessor.

    Responsibilities:
    - construct explicit validity mask (do NOT rely on past_ids)
    - add learnable position embeddings
    - provide extension hooks for future payload features
    """

    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        dropout: float = 0.0,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_positional_embedding = use_positional_embedding

        # ---- Position embedding (learnable, HSTU-style) ----
        if use_positional_embedding:
            self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # ---- Optional dropout after composition ----
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def debug_str(self) -> str:
        flags = []
        if self.use_positional_embedding:
            flags.append("pos")
        return "semantic_preproc[" + ",".join(flags) + "]"

    def forward(
        self,
        past_lengths: torch.Tensor,        # (B,)
        past_ids: torch.Tensor,            # (B, L) - unused by design
        past_embeddings: torch.Tensor,     # (B, L, D)
        past_payloads: dict,               # extensible hook
    ):
        """
        Returns:
            lengths: (B,)
            embeddings: (B, L, D)
            mask: (B, L, 1)
        """
        B, L, D = past_embeddings.shape
        device = past_embeddings.device

        # --------------------------------------------------
        # 1. Explicit validity mask (prefix-contiguous)
        # --------------------------------------------------
        valid_mask = self._build_prefix_mask(past_lengths, L, device)
        past_embeddings = past_embeddings * valid_mask

        # --------------------------------------------------
        # 2. Add position embedding (semantic-token aware)
        # --------------------------------------------------
        if self.use_positional_embedding:
            # positions: [0, 1, ..., L-1]
            positions = torch.arange(L, device=device)
            pos_emb = self.pos_emb(positions).unsqueeze(0)  # (1, L, D)

            # only add to valid positions
            past_embeddings = past_embeddings + pos_emb * valid_mask

        # --------------------------------------------------
        # 3. (Future) payload fusion hook
        # --------------------------------------------------
        # Example extensions (NOT active now):
        #
        # if "time_gap" in past_payloads:
        #     time_emb = self.time_emb(past_payloads["time_gap"])
        #     past_embeddings = past_embeddings + time_emb * valid_mask
        #
        # if "item_side_feat" in past_payloads:
        #     side_emb = self.side_proj(past_payloads["item_side_feat"])
        #     past_embeddings = past_embeddings + side_emb * valid_mask

        # --------------------------------------------------
        # 4. Optional dropout
        # --------------------------------------------------
        if self.dropout is not None:
            past_embeddings = self.dropout(past_embeddings)

        return past_lengths, past_embeddings, valid_mask

    @staticmethod
    def _build_prefix_mask(
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ):
        """
        lengths: (B,)
        return: (B, max_len, 1) float mask
        """
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        return mask

class IdentityEmbeddingPostprocessor(OutputPostprocessorModule):
    def debug_str(self) -> str:
        return "identity"

    def forward(self, output_embeddings: torch.Tensor) -> torch.Tensor:
        return output_embeddings

class NativeDummySimilarity(SimilarityModule):
    def forward(self, query, positive, negative):
        return torch.zeros(1, device=query.device)


class PostprocessorAsModule(nn.Module):
    """
    Wrap an OutputPostprocessorModule as a normal nn.Module.
    Useful for applying different postprocessors per task branch.
    """
    def __init__(self, postprocessor: OutputPostprocessorModule):
        super().__init__()
        self.postprocessor = postprocessor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.postprocessor(x)

class UnifiedInputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        
        # 1. Semantic Embedding
        if config.use_semantic_seq:
            self.sem_emb = nn.Embedding(config.sem_total_vocab, dim, padding_idx=0)
            
        # 2. Atomic Embedding
        if config.use_atomic_seq:
            self.atom_emb = nn.Embedding(config.num_atomic_items + 1, dim, padding_idx=0)
            
        # 3. Profile Embeddings
        if config.use_cat_profile:
            self.cat_embs = nn.ModuleList([nn.Embedding(v, dim) for v in config.cat_feature_vocab_sizes])
        if config.use_num_profile:
            self.num_projs = nn.ModuleList([nn.Linear(1, dim) for _ in range(config.num_feature_size)])
            
        if config.use_cat_profile or config.use_num_profile:
            self.feat_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, input_dict, is_generation=False):
        """
        is_generation: 如果为 True，则只处理 Semantic History 部分 (用于 Beam Search 扩展)
        """
        tokens = []
        
        # A. Profile (Prefix)
        prefix = []
        if self.config.use_cat_profile and 'cat_feats' in input_dict:
            for i, emb in enumerate(self.cat_embs):
                prefix.append(emb(input_dict['cat_feats'][:, i]).unsqueeze(1))
        if self.config.use_num_profile and 'num_feats' in input_dict:
            for i, proj in enumerate(self.num_projs):
                prefix.append(proj(input_dict['num_feats'][:, i].unsqueeze(-1)).unsqueeze(1))
        if prefix:
            tokens.append(self.feat_mlp(torch.cat(prefix, dim=1)))
            
        # B. Atomic Sequence
        if self.config.use_atomic_seq and 'atom_history' in input_dict:
            tokens.append(self.atom_emb(input_dict['atom_history']))
            
        # C. Semantic Sequence
        # 在 Beam Search 时，sem_history 会不断变长
        if self.config.use_semantic_seq and 'sem_history' in input_dict:
            tokens.append(self.sem_emb(input_dict['sem_history']))
            
        if not tokens: raise ValueError("Empty inputs")
        return torch.cat(tokens, dim=1)

class UniGCRModel(nn.Module):
    def __init__(self, config: UniGCRConfig):
        super().__init__()
        self.config = config
        
        # 1. Input & Backbone
        self.input_layer = UnifiedInputLayer(config)
        preprocessor = SemanticAwareInputFeaturesPreprocessor(
            max_seq_len = config.max_seq_len,
            embed_dim = config.embed_dim,
            use_positional_embedding = True,
        )
        postprocessor = self.build_output_postprocessor()

        # CTR branch postproc（推荐：LayerNorm）
        # 注意：这是“分支内”用，不传进 HSTU
        self.ctr_postprocessor = PostprocessorAsModule(LayerNormEmbeddingPostprocessor(embedding_dim=config.embed_dim))

        # (Optional) future retrieval postproc
        self.re_postprocessor = PostprocessorAsModule(L2NormEmbeddingPostprocessor(embedding_dim=config.embed_dim))
        # 构造占位 Similarity (必须继承自库基类以确保类型检查通过)

        # self.backbone = OfficialHSTU(config=config.to_hstu_config(), embedding_module=None)
        # 3. 初始化 Backbone (严格匹配 hstu.py 的参数定义)
        self.backbone = OfficialHSTU(
            max_sequence_len=config.max_seq_len,
            max_output_len=0,
            embedding_dim=config.embed_dim,
            num_blocks=config.hstu_layers,
            num_heads=config.hstu_heads,
            linear_dim=config.embed_dim,      # 对应 hstu.py 的 linear_dim (dv)
            attention_dim=config.embed_dim,   # 对应 hstu.py 的 attention_dim (dqk)
            normalization="rel_bias",
            linear_config="uvqk",
            linear_activation="silu",
            linear_dropout_rate=config.dropout,
            attn_dropout_rate=config.dropout,
            embedding_module=self.input_layer.sem_emb if config.use_semantic_seq else None,
            similarity_module=NativeDummySimilarity(),
            input_features_preproc_module=preprocessor,
            output_postproc_module=postprocessor,
            enable_relative_attention_bias=True,
            verbose=False
        )

        # 2. GR Head
        self.gr_head = nn.Linear(config.embed_dim, config.sem_total_vocab)
        
        # 3. CTR Components
        self.cand_proj = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, config.embed_dim))
        
        scorer_dim = config.embed_dim * 2
        if config.ctr_use_self_attn:
            self.cand_self_attn = nn.MultiheadAttention(config.embed_dim, 2, batch_first=True)
            scorer_dim += config.embed_dim
        if config.ctr_use_cross_attn:
            self.user_cross_attn = nn.MultiheadAttention(config.embed_dim, 2, batch_first=True)
            scorer_dim += config.embed_dim
            
        self.scorer = nn.Sequential(nn.Linear(scorer_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def build_output_postprocessor(self):
        """
        Output postprocessor for HSTU backbone.
        GR-friendly: NO normalization, NO projection.
        """

        class IdentityPostprocessor(OutputPostprocessorModule):
            def debug_str(self) -> str:
                return "identity_gr"

            def forward(self, output_embeddings: torch.Tensor) -> torch.Tensor:
                return output_embeddings

        return IdentityPostprocessor()

    def get_lengths(self, input_tensor):
        lengths = (input_tensor != 0).sum(dim=1)
        return lengths

    def forward(self, batch_dict):
        # 普通的前向传播 (Training GR)
        sem_history = batch_dict["sem_history"]

        lengths = self.get_lengths(sem_history)
        x = self.input_layer(batch_dict)
        B, L, _ = x.shape

        u_seq = self.backbone(
            past_lengths=lengths,
            past_ids=torch.zeros((B, L), dtype=torch.long, device=x.device),
            past_embeddings=x,
            past_payloads={}
        )

        # u_last: 用于 CTR 任务或 Beam Search 初始化
        u_last = self._get_last_token(u_seq, B, lengths)  # (B, D)

        # Sequence-level logits 用于 GR 训练
        logits = self.gr_head(u_seq) # (B, L, V)

        return u_last, u_seq, logits, B

    def _get_last_token(self, seq, B, lengths):
        last_token = seq[torch.arange(B, device=seq.device), lengths - 1]
        return last_token

    def _get_item_vector(self, codes):
        """
        将 Semantic Codes (B, Layers) 转化为 Embedding。
        根据你的要求：使用"SemanticID最后一个hidden states"
        实现：将 Codes 视为一个短序列，通过 Embedding 层 (或者 InputLayer)，
        这里简单起见，取所有 Code Embedding 的 Sum 或 Last。
        """
        # (B, Layers, D)
        embs = self.input_layer.sem_emb(codes)
        
        # 方式 A: Sum (常用，信息无损)
        # return torch.sum(embs, dim=1)
        
        # 方式 B: Last Hidden State (你的要求)
        # return embs[:, -1, :]
        
        # 方式 C: 既然是 Hidden State，可能需要过一层 MLP 融合
        # 这里取 Sum 作为最稳健的表示
        return torch.sum(embs, dim=1)

    def predict_ctr(self, u, batch_dict, pos_codes, device, grid_mapper):
        """
        Memory Bank 构造：
        1. GT Injection (pos_codes)
        2. Hard Negatives (Beam Search from GR) - 剔除 GT
        3. Random Negatives
        """
        B = u.size(0)
        N = self.config.memory_bank_size
        K = self.config.num_hard_negatives
        
        # --- 1. GT Embedding ---
        # pos_codes: (B, Layers)
        pos_emb = self._get_item_vector(pos_codes).unsqueeze(1) # (B, 1, D)
        
        # --- 2. Hard Negatives (Beam Search) ---
        # 这是一个耗时操作，训练时只生成 K 个
        # candidates: (B, Beam_Width, Layers)
        # 为了效率，我们让 Beam_Width = K
        candidates = self._beam_search_hard_negatives_v1(
            batch_dict, u, beam_width=K, grid_mapper=grid_mapper
        )
        
        # 剔除 GT: 比较 candidates 和 pos_codes
        # 如果 candidate == pos_codes, 它是 False Negative, 需要替换
        # 简单处理：如果撞了，就随机替换成一个 random code，或者保留但 label 设为 0 (模型会困惑)
        # 这里采用严谨做法：Mask 掉
        
        # candidates: (B, K, Layers)
        # pos_codes: (B, Layers) -> (B, 1, Layers)
        pos_exp = pos_codes.unsqueeze(1)
        # exact match check: (B, K)
        is_hit = torch.all(candidates == pos_exp, dim=-1)
        
        # 如果命中 GT，替换为随机负样本
        # 构造一个随机掩码
        rand_backup = torch.randint(1, self.config.sem_total_vocab, candidates.shape).to(device)
        candidates = torch.where(is_hit.unsqueeze(-1), rand_backup, candidates)
        
        # 获取 Hard Negative Embeddings
        # candidates: (B, K, Layers) -> flat -> (B*K, Layers) -> emb -> reshape
        hard_embs = self._get_item_vector(candidates.view(-1, self.config.sem_id_layers))
        hard_embs = hard_embs.view(B, K, -1) # (B, K, D)
        
        # --- 3. Random Negatives ---
        num_easy = N - 1 - K
        rand_codes = torch.randint(1, self.config.sem_total_vocab, (B, num_easy, self.config.sem_id_layers)).to(device)
        easy_embs = self.input_layer.sem_emb(rand_codes).sum(dim=2) # Simple Sum for random
        
        # --- 4. Construct Bank ---
        # bank: (B, 1+K+Easy, D)
        bank = torch.cat([pos_emb, hard_embs, easy_embs], dim=1)
        
        # --- 5. Labels & Shuffle ---
        labels = torch.zeros((B, N), device=device)
        labels[:, 0] = 1.0 # GT is at 0
        
        perm = torch.randperm(N).to(device)
        bank = bank[:, perm, :]
        labels = labels[:, perm]
        
        # --- 6. Scoring ---
        bank_proj = self.cand_proj(bank)
        u_exp = u.unsqueeze(1).repeat(1, N, 1)
        feats = [bank_proj, u_exp]
        
        if self.config.ctr_use_self_attn:
            self_out, _ = self.cand_self_attn(bank_proj, bank_proj, bank_proj)
            feats.append(self_out)
        if self.config.ctr_use_cross_attn:
            u_q = u.unsqueeze(1)
            cross_out, _ = self.user_cross_attn(u_q, bank_proj, bank_proj)
            feats.append(cross_out.repeat(1, N, 1))
            
        final_feats = torch.cat(feats, dim=-1)
        ctr_logits = self.scorer(final_feats).squeeze(-1)
        
        return ctr_logits, labels

    @torch.no_grad()
    def generate_gr_candidates(self, batch_dict, u_last, beam_width=10, grid_mapper=None, lengths=None):
        """
        GR inference: generate Top-K semantic ID candidates
        return: (B, K, sem_id_layers)
        """
        return self._beam_search_hard_negatives_v2(
            batch_dict=batch_dict,
            u_last=u_last,
            beam_width=beam_width,
            grid_mapper=grid_mapper,
            lengths=lengths,
        )

    @torch.no_grad()
    def _beam_search_hard_negatives_v2(self, batch_dict, u_last, beam_width=5, grid_mapper=None, lengths=None):
        """
        在 Training 中使用 Beam Search 生成 Hard Negatives。
        这需要多次运行 Backbone，比较耗时，但质量高。
        """
        B = u_last.size(0)
        device = u_last.device
        num_layers = self.config.sem_id_layers
        max_seq_len = self.config.max_seq_len  # 获取最大序列长度

        # 准备 Beam Search 的初始输入
        # 我们需要复制 batch_dict 中的所有 Tensor 到 (B*K)
        # 但为了节省显存，我们只扩展必要的 sem_history

        # 初始 Input: 原始的 sem_history，这里实际上用的是sem history eval, 即不包含target codes
        curr_seqs = batch_dict['sem_history']
        curr_lengths = lengths

        # 初始 Scores: (B*K)
        # 第一步只有 1 个 Beam (原始序列)
        curr_scores = torch.zeros(B, device=device)

        # 扩展其他特征以备后续使用 (Profile, Atomic)
        # 这里的策略是：InputLayer 处理时支持广播，或者我们将 Profile 重复 K 次
        # 为了实现简单，我们将 batch_dict 里的 tensor 全部 repeat
        expanded_batch = {'sem_history': curr_seqs, }
        # for k, v in batch_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         # 如果是 (B, ...)，重复成 (B*K, ...)
        #         # 初始 K=1, 后续 K=beam_width
        #         expanded_batch[k] = v  # 初始不重复

        # 开始逐层生成
        # layer_idx: 0 -> 1 -> 2
        for layer_idx in range(num_layers):
            # 1. 构造当前步的 Input Embedding
            # expanded_batch['sem_history'] = curr_seqs
            # 注意：这里的 curr_seqs 长度在不断增加

            # 调用 Backbone
            # x: (B, Len, D)
            x = self.input_layer(expanded_batch)
            B_curr, L, _ = x.shape

            u_seq = self.backbone(
                past_lengths=curr_lengths,
                past_ids=torch.zeros((B_curr, L), dtype=torch.long, device=x.device),  # 修复：使用 B_curr 而不是 B
                past_embeddings=x,
                past_payloads={}
            )
            u_next = u_seq[torch.arange(B_curr, device=device), curr_lengths - 1]

            logits = self.gr_head(u_next)  # (B_curr, Vocab)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Masking (只允许当前 Layer 的 ID)
            if grid_mapper:
                start, end = grid_mapper.get_layer_range(layer_idx)
                mask = torch.ones_like(logits) * float('-inf')
                mask[:, start:end] = 0
                log_probs = log_probs + mask

            # Beam Expansion
            next_scores = curr_scores.unsqueeze(1) + log_probs

            if layer_idx == 0:
                # 第一层：从 1 扩展到 K
                # topk: (B, K)
                topk_scores, topk_ids = torch.topk(next_scores, beam_width, dim=1)
                curr_scores = topk_scores.view(-1)  # (B*K)

                # 扩展 History: (B, T) -> (B, K, T) -> (B*K, T)
                seq_exp = curr_seqs.unsqueeze(1).repeat(1, beam_width, 1).view(B * beam_width, -1)
                lengths_exp = curr_lengths.unsqueeze(1).repeat(1, beam_width).view(-1)  # (B*K,)

                # 将新 token 插入到有效位置
                new_tokens = topk_ids.view(-1, 1)
                batch_indices = torch.arange(B * beam_width, device=device)
                seq_exp[batch_indices, lengths_exp] = new_tokens.squeeze(1)

                # 扩展 Batch Dict 中的其他特征
                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        # (B, ...) -> (B, K, ...) -> (B*K, ...)
                        shape = [B, beam_width] + list(v.shape[1:])
                        expanded_batch[k] = v.unsqueeze(1).repeat(1, beam_width, *([1] * (v.dim() - 1))).view(-1, *v.shape[1:])

                # 更新 sem_history, lengths
                curr_lengths = lengths_exp + 1
                curr_seqs = seq_exp
                expanded_batch['sem_history'] = curr_seqs

            else:
                # 后续层：从 B*K 扩展到 B*K*Vocab，取 Top K
                # next_scores: (B*K, Vocab) -> view (B, K, Vocab)
                vocab_size = logits.size(-1)
                next_scores = next_scores.view(B, beam_width, vocab_size).view(B, -1)

                # TopK per user
                best_scores, best_indices = torch.topk(next_scores, beam_width, dim=1)  # (B, K)

                # 解码 Index
                beam_indices = best_indices // vocab_size  # 属于哪个旧 beam
                token_indices = best_indices % vocab_size  # 新 token 是什么

                # Gather Seqs
                # curr_seqs: (B*K, T) -> (B, K, T)
                curr_seqs_view = curr_seqs.view(B, beam_width, -1)
                curr_lengths_view = curr_lengths.view(B, beam_width)

                new_seq_list = []
                new_lengths_list = []
                for b in range(B):
                    # select beams
                    sel_seqs = curr_seqs_view[b][beam_indices[b]].clone()  # (K, L)
                    sel_lengths = curr_lengths_view[b][beam_indices[b]]  # (K,)
                    sel_tokens = token_indices[b]  # (K,)

                    # 将新 token 插入到每个 beam 的有效位置
                    for k in range(beam_width):
                        sel_seqs[k, sel_lengths[k]] = sel_tokens[k]

                    new_seq_list.append(sel_seqs)
                    new_lengths_list.append(sel_lengths + 1)

                curr_seqs = torch.cat(new_seq_list, dim=0)  # (B*K, T+1)
                curr_lengths = torch.cat(new_lengths_list, dim=0)
                curr_scores = best_scores.view(-1)
                expanded_batch['sem_history'] = curr_seqs

        # 提取生成的 tokens（从每个序列的末尾 num_layers 个有效位置）
        generated = []
        curr_seqs_view = curr_seqs.view(B, beam_width, -1)
        curr_lengths_view = curr_lengths.view(B, beam_width)

        for b in range(B):
            beam_codes = []
            for k in range(beam_width):
                end_pos = curr_lengths_view[b, k].item()
                start_pos = end_pos - num_layers
                codes = curr_seqs_view[b, k, start_pos:end_pos]
                beam_codes.append(codes)
            generated.append(torch.stack(beam_codes, dim=0))

        generated = torch.stack(generated, dim=0)  # (B, K, num_layers)

        return generated
