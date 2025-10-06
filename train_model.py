#!/usr/bin/env python3
"""
DeepGuard Training Script - SIMPLIFIED WORKING VERSION
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import json

# Simple dataset without complex operations
class SimpleDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Could not load image: {self.image_paths[idx]}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
            
        except Exception as e:
            print(f"Error with {self.image_paths[idx]}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(self.labels[idx], dtype=torch.long)

# Simple model
class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'data': {'batch_size': 16, 'test_size': 0.2},
            'training': {'num_epochs': 10, 'learning_rate': 0.0001},
            'model': {'num_classes': 2}
        }

def prepare_data(real_dir, fake_dir, test_size=0.2):
    real_paths = glob.glob(os.path.join(real_dir, '*.jpg')) + \
                 glob.glob(os.path.join(real_dir, '*.png')) + \
                 glob.glob(os.path.join(real_dir, '*.jpeg'))
    
    fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg')) + \
                 glob.glob(os.path.join(fake_dir, '*.png')) + \
                 glob.glob(os.path.join(fake_dir, '*.jpeg'))
    
    # Create dummy data if no images found
    if not real_paths or not fake_paths:
        print("Creating sample images...")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        for i in range(5):
            # Real images
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(f'{real_dir}/real_{i}.jpg', img)
            # Fake images
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(f'{fake_dir}/fake_{i}.jpg', img)
        
        real_paths = glob.glob(os.path.join(real_dir, '*.jpg'))
        fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg'))
    
    real_train, real_temp = train_test_split(real_paths, test_size=test_size*2, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
    
    fake_train, fake_temp = train_test_split(fake_paths, test_size=test_size*2, random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)
    
    return (real_train, fake_train), (real_val, fake_val), (real_test, fake_test)

class DeepfakeTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return total_loss / len(self.val_loader), accuracy, precision, recall, f1
    
    def train(self, num_epochs: int = 10):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}")
            print(f"Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
            print("-" * 50)
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': self.best_accuracy,
                }, 'models/best_model.pth')
                print(f"✓ Saved best model with accuracy: {val_acc:.4f}")
    
    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return all_preds, all_labels, all_probs
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("✓ Training history plot saved")

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override config
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    (real_train, fake_train), (real_val, fake_val), (real_test, fake_test) = prepare_data(
        'data/real_images', 'data/fake_images', config['data']['test_size']
    )

    print(f"Training: {len(real_train)} real, {len(fake_train)} fake")
    print(f"Validation: {len(real_val)} real, {len(fake_val)} fake")
    print(f"Test: {len(real_test)} real, {len(fake_test)} fake")

    # Initialize model - USING SIMPLE VERSION
    model = SimpleDetector(num_classes=config['model']['num_classes'])
    
    # Create datasets
    train_dataset = SimpleDataset(real_train + fake_train, [0]*len(real_train) + [1]*len(fake_train))
    val_dataset = SimpleDataset(real_val + fake_val, [0]*len(real_val) + [1]*len(fake_val))
    test_dataset = SimpleDataset(real_test + fake_test, [0]*len(real_test) + [1]*len(fake_test))
    
    # Create data loaders
    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize trainer
    trainer = DeepfakeTrainer(model, device, train_loader, val_loader, test_loader)
    
    # Train the model
    trainer.train(config['training']['num_epochs'])
    
    # Test the model
    preds, labels, probs = trainer.test()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1)
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("✓ Training completed! Results saved to training_results.json")

if __name__ == "__main__":
    main()