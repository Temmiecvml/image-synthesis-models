import torch
import torch.nn as nn
from clip import clip


class CLIPModel(nn.Module):
    def __init__(self, clip_text_model, device):
        super(CLIPModel, self).__init__()
        self.model, _ = clip.load(clip_text_model, device)

    def forward(self, text):
        device = self.get_device()
        query = [text] if not hasattr(text, "__iter__") else text
        query_tokens = torch.vstack([clip.tokenize([q]) for q in query]).to(device)
        with torch.no_grad():
            query = self.model.encode_text(query_tokens, embeddings=True)

        return query

    def get_device(self):
        for param in self.parameters():
            return param.device


class TextConditioner(nn.Module):
    def __init__(
        self,
        dims,
        out_dims,
        num_heads,
        device,
        clip_dtype,
        clip_model="ViT-B/32",
    ):
        super(TextConditioner, self).__init__()

        self.load_clip(clip_model, device, clip_dtype)

        self.norm = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(dims, num_heads, batch_first=True)
        self.projection = nn.Linear(dims, out_dims)

    def load_clip(self, clip_model, device, clip_dtype):
        clip = CLIPModel(clip_model, device)

        for param in clip.parameters():
            param.data = param.data.to(clip_dtype)
        for buffer in clip.buffers():
            buffer.data = buffer.data.to(clip_dtype)

        self.clip = clip

    def forward(self, query):
        with torch.no_grad():
            q = self.clip(query)

        q = self.norm(q)
        attn_output, _ = self.attn(q, q, q)
        o = self.projection(attn_output)

        return o
