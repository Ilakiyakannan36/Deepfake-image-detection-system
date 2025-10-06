import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class DeepfakeTrainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for images, forensic_features, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            forensic_features = forensic_features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, forensic_features)
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
            for images, forensic_features, labels in self.val_loader:
                images = images.to(self.device)
                forensic_features = forensic_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, forensic_features)
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
    
    def train(self, num_epochs: int = 20):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}")
            print(f"Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
            print("-" * 50)
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': self.best_accuracy,
                }, 'models/best_model.pth')
                print(f"Saved best model with accuracy: {val_acc:.4f}")
    
    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, forensic_features, labels in self.test_loader:
                images = images.to(self.device)
                forensic_features = forensic_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, forensic_features)
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