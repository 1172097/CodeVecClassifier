import json
import torch
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from Embedding import VocabularyBuilder
from Classifier import ImprovedCodeClassifier
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Embedding import VocabularyBuilder, collate_path_contexts
from AST import main as ast_main


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            logits, _ = model(batch['start_tokens'], batch['paths'], 
                            batch['end_tokens'], batch['mask'])
            
            loss = criterion(logits, batch['label'])
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Validation phase
        val_loss, accuracy = validate_model(model, val_loader, criterion, device)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%\n')
    
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, _ = model(batch['start_tokens'], batch['paths'], 
                            batch['end_tokens'], batch['mask'])
            
            loss = criterion(logits, batch['label'])
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()
    
    return total_loss/len(val_loader), 100 * correct/total
# Dataset class for code samples
class CodeDataset(Dataset):
    def __init__(self, path_contexts_list, labels, vocab_builder, max_contexts=200):
        self.path_contexts_list = path_contexts_list
        self.labels = labels
        self.vocab_builder = vocab_builder
        self.max_contexts = max_contexts
        
    def __len__(self):
        return len(self.path_contexts_list)
        
    def __getitem__(self, idx):
        contexts = self.path_contexts_list[idx]
        label = self.labels[idx]
        
        # Convert to tensors using collate function
        batch_data = collate_path_contexts([contexts], self.vocab_builder, self.max_contexts)
        
        return {
            'start_tokens': batch_data['start_tokens'][0],
            'paths': batch_data['paths'][0],
            'end_tokens': batch_data['end_tokens'][0],
            'mask': batch_data['mask'][0],
            'label': torch.tensor(label, dtype=torch.long)
        }



# Assuming your code file is named 'model.py' or similar


def load_and_preprocess_data(file_path):
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract contexts and labels
    path_contexts_list = [item['contexts'] for item in data]
    labels = [item['label'] for item in data]
    
    # Create label to index mapping
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert string labels to indices
    label_indices = [label_to_idx[label] for label in labels]
    
    return path_contexts_list, label_indices, label_to_idx

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # File paths
    data_path = './training_data.json'
    save_dir = os.path.dirname(data_path)
    
    # Load and preprocess data
    path_contexts_list, label_indices, label_to_idx = load_and_preprocess_data(data_path)
    
    # Split data into train and validation sets
    train_contexts, val_contexts, train_labels, val_labels = train_test_split(
        path_contexts_list, 
        label_indices,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize vocabulary builder
    vocab_builder = VocabularyBuilder()
    
    # Build vocabulary from training data
    for contexts in train_contexts:
        vocab_builder.build_vocab(contexts)
    
    # Create datasets
    train_dataset = CodeDataset(train_contexts, train_labels, vocab_builder)
    val_dataset = CodeDataset(val_contexts, val_labels, vocab_builder)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = ImprovedCodeClassifier(
        token_vocab_size=len(vocab_builder.token_to_idx),
        path_vocab_size=len(vocab_builder.path_to_idx),
        num_classes=len(label_to_idx),
        embedding_dim=256,
        num_heads=8,
        num_layers=3
    )
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001,
        device=device
    )
    
    # Save model and vocabularies
    model_save_path = os.path.join(save_dir, 'code_classifier.pt')
    vocab_save_path = os.path.join(save_dir, 'vocab_data.pt')
    label_map_save_path = os.path.join(save_dir, 'label_map.json')
    
    # Save model state
    torch.save(trained_model.state_dict(), model_save_path)
    
    # Save vocabulary data
    torch.save({
        'token_to_idx': vocab_builder.token_to_idx,
        'path_to_idx': vocab_builder.path_to_idx
    }, vocab_save_path)
    
    # Save label mapping
    with open(label_map_save_path, 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_save_path}")
    print(f"Vocabulary data saved to: {vocab_save_path}")
    print(f"Label mapping saved to: {label_map_save_path}")

if __name__ == "__main__":
    # First run AST processing
    ast_main()
    # Then run the model training
    main()
