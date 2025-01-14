import torch.nn as nn
from Embedding import CodeEmbedding

class ImprovedCodeClassifier(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        path_vocab_size: int,
        num_classes: int,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # Code embedding component
        self.code_embedding = CodeEmbedding(
            token_vocab_size=token_vocab_size,
            path_vocab_size=path_vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )
        
    def forward(self, start_tokens, paths, end_tokens, mask=None):
        code_vector, attention_weights = self.code_embedding(
            start_tokens, paths, end_tokens, mask
        )
        
        # Apply transformer
        transformed = self.transformer(code_vector.unsqueeze(1))
        
        # Get class logits
        logits = self.classifier(transformed.squeeze(1))
        
        return logits, attention_weights

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)
