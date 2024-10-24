import torch
import torch.nn as nn


class CLIP(nn.Module):
    def __init__(self, clip_model, clip_text_model):
        super(CLIP, self).__init__()
        self.clip_model = clip_model
        self.clip_text_model = clip_text_model

    def forward(self, x, text):
        x = self.clip_model.encode_image(x)
        text = self.clip_text_model.encode_text(text)
        return x, text
