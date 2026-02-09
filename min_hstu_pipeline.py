import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from research_hstu.modeling.sequential.hstu import HSTU
import os

class SimpleEmbeddingModule(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.item_embedding_dim = embed_dim
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(item_ids)


class IdentityInputPreprocessor(nn.Module):
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ):
        return past_lengths, past_embeddings, None


class IdentityOutputPostprocessor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class DummySimilarityModule(nn.Module):
    def forward(self, x, y):
        raise NotImplementedError

def load_grid_user_sequences(pt_path: str):
    """
    返回:
        user2seq: Dict[int, List[int]]
    """
    user2seq = torch.load(pt_path, map_location="cpu")
    assert isinstance(user2seq, dict)
    return user2seq

def collate_grid_batch(
    user_ids,
    user2seq,
    max_seq_len,
    pad_token=0,
    device="cuda",
):
    """
    返回:
        sem_history: [B, L]
        lengths: [B]
    """
    seqs = []
    lengths = []

    for uid in user_ids:
        seq = user2seq[uid]

        # 截断（保留最近）
        seq = seq[-max_seq_len:]
        lengths.append(len(seq))

        # 左 padding
        if len(seq) < max_seq_len:
            seq = [pad_token] * (max_seq_len - len(seq)) + seq

        seqs.append(seq)

    sem_history = torch.tensor(seqs, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    return sem_history, lengths


def collate_grid_user_sequences(
    user_ids: List[int],
    user2seq: Dict[int, List[int]],
    max_seq_len: int,
    pad_token: int = 0,
    device: str = "cuda",
):
    """
    将 GRID 的 user -> semantic token list
    转为 HSTU 可用的 [B, L] + lengths
    """
    seqs = []
    lengths = []

    for uid in user_ids:
        seq = user2seq[uid]

        # 只保留最近 max_seq_len 个
        seq = seq[-max_seq_len:]
        lengths.append(len(seq))

        # 左侧 padding
        if len(seq) < max_seq_len:
            seq = [pad_token] * (max_seq_len - len(seq)) + seq

        seqs.append(seq)

    sem_history = torch.tensor(seqs, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    return sem_history, lengths


# ============================================================
# 4. Minimal HSTU Model（只微改 forward 接 lengths）
# ============================================================

class MinimalHSTUModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()

        self.device = device

        self.embedding = SimpleEmbeddingModule(vocab_size, embed_dim)
        self.preproc = IdentityInputPreprocessor()
        self.postproc = IdentityOutputPostprocessor()
        self.similarity = DummySimilarityModule()

        self.hstu = HSTU(
            max_sequence_len=max_seq_len,
            max_output_len=0,
            embedding_dim=embed_dim,
            num_blocks=num_layers,
            num_heads=num_heads,
            linear_dim=embed_dim // num_heads,
            attention_dim=embed_dim // num_heads,
            normalization="rel_bias",
            linear_config="uvqk",
            linear_activation="silu",
            linear_dropout_rate=dropout,
            attn_dropout_rate=dropout,
            embedding_module=self.embedding,
            similarity_module=self.similarity,
            input_features_preproc_module=self.preproc,
            output_postproc_module=self.postproc,
            enable_relative_attention_bias=True,
            concat_ua=False,
            verbose=False,
        )

        self.token_decoder = nn.Linear(embed_dim, vocab_size)
        self.to(device)

    def forward(
        self,
        sem_history: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sem_history: [B, L]
        lengths: [B]
        """
        embeds = self.embedding.get_item_embeddings(sem_history)

        seq_embeds, _ = self.hstu.generate_user_embeddings(
            past_lengths=lengths,
            past_ids=sem_history,
            past_embeddings=embeds,
            past_payloads={},  # 不用 timestamp
        )

        user_embeds = seq_embeds[
            torch.arange(len(lengths), device=lengths.device),
            lengths - 1,
        ]

        logits = self.token_decoder(user_embeds)
        return user_embeds, logits

    @torch.no_grad()
    def greedy_generate(
        self,
        sem_history: torch.Tensor,
        lengths: torch.Tensor,
        num_tokens: int = 3,
    ):
        cur_history = sem_history
        cur_lengths = lengths.clone()
        generated = []

        for _ in range(num_tokens):
            _, logits = self.forward(cur_history, cur_lengths)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)

            cur_history = torch.cat([cur_history, next_token], dim=1)
            cur_lengths += 1

        return torch.cat(generated, dim=1)

def main():
    print("=== Running HSTU with GRID semantic sequences ===")
    grid_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GRID')
    GRID_PT = os.path.join(grid_dir, "outputs/beauty_user_semantic_sequences_training.pt")

    # 1. load GRID data
    user2seq = load_grid_user_sequences(GRID_PT)

    # 2. sample 一个 batch（示例）
    user_ids = list(user2seq.keys())[:8]  # batch size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_MAX_LEN = 30
    GEN_LEN = 3
    sem_history, lengths = collate_grid_batch(
        user_ids=user_ids,
        user2seq=user2seq,
        max_seq_len=DATA_MAX_LEN,
        device=device,
    )
    vocab_size = 500        # 必须 >= GRID 最大 semantic id + 1
    embed_dim = 64
    max_seq_len = DATA_MAX_LEN + GEN_LEN
    num_layers = 2
    num_heads = 4
    dropout = 0.1

    model = MinimalHSTUModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        device=device,
    )
    model.eval()

    with torch.no_grad():
        user_embeds, logits = model(sem_history, lengths)

    print("User embeddings:", user_embeds.shape)
    print("Logits:", logits.shape)

    with torch.no_grad():
        gen_tokens = model.greedy_generate(
            sem_history, lengths, num_tokens=3
        )

    print("Generated semantic tokens:")
    print(gen_tokens)

    print("=== SUCCESS ===")


# ============================================================
# 6. Example usage
# ============================================================

if __name__ == "__main__":

    main()

