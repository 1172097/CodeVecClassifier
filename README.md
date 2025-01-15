# CodeVecClassifier

CodeVecClassifier is a deep learning model for classifying algorithms based on their source code, inspired by the code2vec paper. It uses path-based representations of code to generate embeddings and perform classification.

## Architecture

The system consists of several key components:

- **Code Embedding**: Utilizes token and path embeddings with attention mechanism
- **AST Processing**: Extracts path contexts from source code using Abstract Syntax Trees
- **Classifier**: An improved transformer-based classifier with multi-head attention
- **Vocabulary Builder**: Handles token and path vocabulary management

### Model Features

- Token and path embeddings with dimension of 256
- Multi-head attention with 8 heads
- 3 transformer layers
- AdamW optimizer with OneCycleLR scheduler
- Early stopping mechanism
- Gradient clipping for stable training

## Installation

```bash
pip install torch
# Add other dependencies as needed
```

## Usage

### Training

```python
from main import main

# Run AST processing and model training
if __name__ == "__main__":
    main()
```

### Prediction

### Prediction

```python
from predict import CodePredictor

predictor = CodePredictor()
code = """
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

predicted_label, confidence, attention_weights = predictor.predict(code)
print(f"Predicted Algorithm: {predicted_label}")
print(f"Confidence: {confidence:.2%}")
```

Example output:

```
Predicted Algorithm: Binary Search
Confidence: 98.76%
```

## Model Training

The model is trained using:

- Training/validation split: 80/20
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.01
- Maximum contexts per sample: 200

## References

This implementation is inspired by:

- [code2vec: Learning Distributed Representations of Code](https://arxiv.org/abs/1803.09473)
