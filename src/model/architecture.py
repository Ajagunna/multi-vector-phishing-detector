"""
Multi-Vector Phishing Classifier
Matches Section 3.9.2 / 3.9.3 of the project report.
Fuses contextual text embedding with metadata features.
Uses a compact surrogate encoder for sandbox-friendly training.
"""

import torch
import torch.nn as nn
from typing import Optional

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
NUM_METADATA_FEATURES = 6
HIDDEN_SIZE = 128  # compact for demo; production DistilBERT uses 768


class MultiVectorPhishingClassifier(nn.Module):
    """
    Multi-vector model:
      Compact text encoder -> [CLS] embedding
      + BatchNorm(metadata)
      -> concat -> MLP head -> 2-class logits
    """

    def __init__(
        self,
        meta_dim: int = NUM_METADATA_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        use_surrogate: bool = True,
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.hidden_size = hidden_size
        self.use_surrogate = use_surrogate

        if use_surrogate:
            vocab_size = 260  # char-level
            self.token_emb = nn.Embedding(vocab_size, hidden_size)
            self.pos_emb = nn.Embedding(MAX_LENGTH, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        else:
            from transformers import AutoModel
            self.transformer = AutoModel.from_pretrained(MODEL_NAME)
            hidden_size = self.transformer.config.hidden_size
            self.hidden_size = hidden_size

        self.meta_bn = nn.BatchNorm1d(meta_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + meta_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_surrogate:
            B, L = input_ids.shape
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            x = self.token_emb(input_ids.clamp(0, 259)) + self.pos_emb(positions)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            pad_mask = (attention_mask == 0)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
            full_mask = torch.cat([cls_mask, pad_mask], dim=1)
            out = self.encoder(x, src_key_padding_mask=full_mask)
            return out[:, 0, :]
        else:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata_features: torch.Tensor,
    ) -> torch.Tensor:
        cls_emb = self.encode_text(input_ids, attention_mask)
        # BatchNorm is unreliable with batch size 1 in eval (running stats can be cold).
        # Skip normalization for single-sample inference to keep hybrid scoring stable.
        if metadata_features.size(0) == 1:
            normalized_meta = metadata_features
        else:
            normalized_meta = self.meta_bn(metadata_features)
        fused = torch.cat([cls_emb, normalized_meta], dim=1)
        return self.classifier(fused)


def build_model(device: Optional[torch.device] = None, use_surrogate: bool = True) -> MultiVectorPhishingClassifier:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiVectorPhishingClassifier(use_surrogate=use_surrogate)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = build_model()
    print("[INFO] Model:", model.__class__.__name__)
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ids = torch.randint(0, 100, (2, 32))
    mask = torch.ones(2, 32)
    meta = torch.randn(2, 6)
    logits = model(ids, mask, meta)
    print("       Forward OK, logits:", logits.shape)
