import torch
import torch.nn as nn

from clip import clip


class CLIPModel(nn.Module):
    def __init__(self, clip_text_model, device):
        super(CLIPModel, self).__init__()
        self.model, _ = clip.load(clip_text_model, device=device)

    def forward(self, text, device):
        query = [text] if not hasattr(text, "__iter__") else text
        query_tokens = torch.vstack([clip.tokenize([q]) for q in query]).to(device)
        with torch.no_grad():
            query = self.model.encode_text(query_tokens, embeddings=True)

        return query


class TextConditioner(nn.Module):
    def __init__(self, dims, out_dims, num_heads, clip_model="ViT-B/32", device=None):
        super(TextConditioner, self).__init__()
        self.device = device
        self.clip = CLIPModel(clip_model, device=self.device)
        self.norm = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(
            dims, num_heads, batch_first=True, device=self.device
        )
        self.projection = nn.Linear(dims, out_dims)
        self.to(device)

    def forward(self, query):
        with torch.no_grad():
            q = self.clip(query, device=self.device)

        q = self.norm(q)
        attn_output, _ = self.attn(q, q, q)
        o = self.projection(attn_output)

        return o
