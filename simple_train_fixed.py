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

# Simple dataset without complex OpenCV operations
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
            # Read image with OpenCV
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Could not load image: {self.image_paths[idx]}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            image = self.transform(image)
            
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
            
        except Exception as e:
            print(f"Error with image {self.image_paths[idx]}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, torch.tensor(self.labels[idx], dtype=torch.long)

# Simple model without forensic features
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
            'training': {'num_epochs': 10, 'learning_rate': 0.0001}
        }

def main():
    print("🚀 Starting Simplified Deepfake Training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config()
    
    # Find images
    real_dir = 'data/real_images'
    fake_dir = 'data/fake_images'
    
    real_paths = glob.glob(os.path.join(real_dir, '*.jpg')) + glob.glob(os.path.join(real_dir, '*.png')) + glob.glob(os.path.join(real_dir, '*.jpeg'))
    fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg')) + glob.glob(os.path.join(fake_dir, '*.png')) + glob.glob(os.path.join(fake_dir, '*.jpeg'))
    
    print(f"Found {len(real_paths)} real images and {len(fake_paths)} fake images")
    
    if not real_paths or not fake_paths:
        print("Creating sample images for testing...")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        for i in range(10):
            # Real images (more natural)
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(f'{real_dir}/real_sample_{i}.jpg', img)
            
            # Fake images (more artificial)
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            cv2.imwrite(f'{fake_dir}/fake_sample_{i}.jpg', img)
        
        real_paths = glob.glob(os.path.join(real_dir, '*.jpg'))
        fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg'))
        print(f"Created {len(real_paths)} sample real and {len(fake_paths)} sample fake images")
    
    # Prepare data
    all_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    # Split data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, labels, test_size=config['data']['test_size'], random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"Training: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create datasets
    train_dataset = SimpleDataset(train_paths, train_labels)
    val_dataset = SimpleDataset(val_paths, val_labels)
    test_dataset = SimpleDataset(test_paths, test_labels)
    
    # Create data loaders
    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model, loss, optimizer
    model = SimpleDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_accuracy = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_simple_model.pth')
            print(f"✓ Saved best model with accuracy: {val_accuracy:.2f}%")
    
    # Final test
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    test_precision = precision_score(test_true, test_preds, zero_division=0)
    test_recall = recall_score(test_true, test_preds, zero_division=0)
    test_f1 = f1_score(test_true, test_preds, zero_division=0)
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1)
    }
    
    import json
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("✅ Training completed successfully!")
    print("💾 Model saved as 'models/best_simple_model.pth'")
    print("📊 Results saved as 'training_results.json'")

if __name__ == "__main__":
    main()