import torch
import json
from Embedding import VocabularyBuilder, collate_path_contexts
from Classifier import ImprovedCodeClassifier
from AST import PathContextExtractor


class CodePredictor:
    def __init__(self, model_dir='.'):
        """Initialize CodePredictor with model and vocabularies."""
        with open('./algorithm_analysis/label_map.json', 'r') as f:
            self.label_to_idx = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Load vocabulary data
        vocab_data = torch.load('./algorithm_analysis/vocab_data.pt', weights_only=True)
        self.vocab_builder = VocabularyBuilder()
        self.vocab_builder.token_to_idx = vocab_data['token_to_idx']
        self.vocab_builder.path_to_idx = vocab_data['path_to_idx']
        
        # Initialize improved model
        self.model = ImprovedCodeClassifier(
            token_vocab_size=len(self.vocab_builder.token_to_idx),
            path_vocab_size=len(self.vocab_builder.path_to_idx),
            num_classes=len(self.label_to_idx),
            embedding_dim=256,
            num_heads=8,
            num_layers=3
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load('./algorithm_analysis/code_classifier.pt', weights_only=True))
        self.model.eval()

    def predict(self, code):
        """Predict the algorithm for a given list of contexts."""
        # Prepare input data
        extractor = PathContextExtractor()
        contexts = extractor.extract_path_contexts_for_classification(code)
        batch_data = collate_path_contexts([contexts], self.vocab_builder)
        
        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        start_tokens = batch_data['start_tokens'].to(device)
        paths = batch_data['paths'].to(device)
        end_tokens = batch_data['end_tokens'].to(device)
        mask = batch_data['mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            logits, attention_weights = self.model(start_tokens, paths, end_tokens, mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            
            # Get embeddings for visualization/analysis
            code_vector, _ = self.model.code_embedding(start_tokens, paths, end_tokens, mask)
        
        predicted_label = self.idx_to_label[predicted_idx]
        
        return predicted_label, confidence, attention_weights

# Example usage
if __name__ == "__main__":
    # Example contexts
    test = "def binary_search(arr, target):\n    left = 0\n    right = len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    # Initialize predictor and make prediction
    predictor = CodePredictor()
    predicted_label, confidence, weight = predictor.predict(test)
    
    print(f"Predicted Algorithm: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")

