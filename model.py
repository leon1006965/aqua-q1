import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, num_layers=10, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb

        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln(x)
        logits = self.head(x)
        return logits
