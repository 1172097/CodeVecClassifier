import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns

@dataclass
class PathContext:
    start_token: str
    path: str
    end_token: str

class CodeEmbedding(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        path_vocab_size: int,
        embedding_dim: int = 128,
        dropout_rate: float = 0.25
    ):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(token_vocab_size, embedding_dim)
        self.path_embedding = nn.Embedding(path_vocab_size, embedding_dim)
        
        # Combined context vector transformation
        self.context_transform = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention vector
        self.attention = nn.Parameter(torch.empty(embedding_dim).uniform_(-0.1, 0.1))
        
    def forward(self, 
                start_tokens: torch.Tensor,    # Shape: [batch_size, max_contexts]
                paths: torch.Tensor,           # Shape: [batch_size, max_contexts]
                end_tokens: torch.Tensor,      # Shape: [batch_size, max_contexts]
                mask: torch.Tensor = None):    # Shape: [batch_size, max_contexts]
        
        # Get embeddings for each component
        start_embedded = self.token_embedding(start_tokens)    # [batch_size, max_contexts, embed_dim]
        path_embedded = self.path_embedding(paths)             # [batch_size, max_contexts, embed_dim]
        end_embedded = self.token_embedding(end_tokens)        # [batch_size, max_contexts, embed_dim]
        
        # Concatenate all embeddings
        context_vectors = torch.cat(
            [start_embedded, path_embedded, end_embedded],
            dim=-1
        )  # [batch_size, max_contexts, 3 * embed_dim]
        
        # Transform to get combined context vectors
        combined = self.context_transform(context_vectors)  # [batch_size, max_contexts, embed_dim]
        
        # Calculate attention weights
        attention_weights = torch.matmul(combined, self.attention)  # [batch_size, max_contexts]
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
            
        attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, max_contexts]
        
        # Apply attention to get final code vector
        code_vector = torch.sum(
            combined * attention_weights.unsqueeze(-1),
            dim=1
        )  # [batch_size, embed_dim]
        
        return code_vector, attention_weights

class VocabularyBuilder:
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token_to_idx = {pad_token: 0, unk_token: 1}
        self.path_to_idx = {pad_token: 0, unk_token: 1}
        
    def build_vocab(self, path_contexts: List[PathContext]):
        # Build token vocabulary
        for context in path_contexts:
            self._add_token(context["start_token"])   
            self._add_token(context["end_token"])
            self._add_path(context["path"])
    
    def _add_token(self, token: str):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.token_to_idx)
            
    def _add_path(self, path: str):
        if path not in self.path_to_idx:
            self.path_to_idx[path] = len(self.path_to_idx)
    
    def get_token_index(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
        
    def get_path_index(self, path: str) -> int:
        return self.path_to_idx.get(path, self.path_to_idx[self.unk_token])

def collate_path_contexts(
    path_contexts: List[List[PathContext]],
    vocab: VocabularyBuilder,
    max_contexts: int = 200
) -> Dict[str, torch.Tensor]:
    batch_size = len(path_contexts)
    
    # Initialize tensors
    start_tokens = torch.zeros(batch_size, max_contexts, dtype=torch.long)
    paths = torch.zeros(batch_size, max_contexts, dtype=torch.long)
    end_tokens = torch.zeros(batch_size, max_contexts, dtype=torch.long)
    mask = torch.zeros(batch_size, max_contexts, dtype=torch.bool)
    
    # Fill tensors
    for i, contexts in enumerate(path_contexts):
        contexts = contexts[:max_contexts]  # Limit number of contexts
        for j, context in enumerate(contexts):
            start_tokens[i, j] = vocab.get_token_index(context["start_token"])
            paths[i, j] = vocab.get_path_index(context["path"])
            end_tokens[i, j] = vocab.get_token_index(context["end_token"])
            mask[i, j] = True
            
    return {
        'start_tokens': start_tokens,
        'paths': paths,
        'end_tokens': end_tokens,
        'mask': mask
    }


def compare_code_vectors(
    context_list: List[List[Dict]], 
    model: CodeEmbedding,
    vocab_builder: VocabularyBuilder,
    visualize: bool = True
) -> Dict:
    """
    Compare multiple code snippets using their vector representations
    Args:
        context_list: List of context lists, one for each code snippet
        model: Trained CodeEmbedding model
        vocab_builder: Vocabulary builder instance
        visualize: Whether to show visualization plots
    Returns:
        Dictionary containing vectors and similarity matrix
    """
    # Get embeddings for each code snippet
    vectors = []
    for contexts in context_list:
        batch_data = collate_path_contexts([contexts], vocab_builder)
        with torch.no_grad():
            code_vector, _ = model(
                batch_data['start_tokens'],
                batch_data['paths'],
                batch_data['end_tokens'],
                batch_data['mask']
            )
            vectors.append(code_vector[0].numpy())
    
    vectors = np.stack(vectors)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    if visualize:
        # Plot vectors
        plt.figure(figsize=(10, 6))
        plt.imshow(vectors, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Code Vectors Comparison")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Code Snippets")
        plt.show()

        # Plot similarity matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   cmap='coolwarm',
                   xticklabels=[f"Code {i+1}" for i in range(len(context_list))],
                   yticklabels=[f"Code {i+1}" for i in range(len(context_list))])
        plt.title("Cosine Similarity Matrix")
        plt.show()
    
    return {
        'vectors': vectors,
        'similarity_matrix': similarity_matrix
    }
