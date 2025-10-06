import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple
import glob

class DeepfakeDataset(Dataset):
    def __init__(self, real_paths: List[str], fake_paths: List[str], 
                 transform=None, augment: bool = True):
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.all_paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        self.transform = transform
        self.augment = augment
        
        self.forensic_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.all_paths)
    
    def extract_forensic_features(self, image: np.ndarray) -> torch.Tensor:
        features = []
        
        # Color features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        features.extend(hist_h / (hist_h.sum() + 1e-6))
        features.extend(hist_s / (hist_s.sum() + 1e-6))
        features.extend(hist_v / (hist_v.sum() + 1e-6))
        
        # Noise features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(noise / 1000.0)
        
        return torch.FloatTensor(features)
    
    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        label = self.labels[idx]
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform and self.augment:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                normalized = self.forensic_transform(image=image)
                image = normalized['image']
            
            # Extract forensic features from resized image
            if isinstance(image, torch.Tensor):
                img_for_features = image.permute(1, 2, 0).numpy()
            else:
                img_for_features = image
            
            img_resized = cv2.resize(img_for_features, (224, 224))
            forensic_features = self.extract_forensic_features(img_resized)
            
            return image, forensic_features, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_features = torch.zeros(25)
            return dummy_image, dummy_features, torch.tensor(label, dtype=torch.long)

def get_transforms(augment: bool = True):
    if augment:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def prepare_data(real_dir: str, fake_dir: str, test_size: float = 0.2):
    real_paths = glob.glob(os.path.join(real_dir, '*.jpg')) + \
                 glob.glob(os.path.join(real_dir, '*.png')) + \
                 glob.glob(os.path.join(real_dir, '*.jpeg'))
    
    fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg')) + \
                 glob.glob(os.path.join(fake_dir, '*.png')) + \
                 glob.glob(os.path.join(fake_dir, '*.jpeg'))
    
    # Create dummy data if no images found
    if not real_paths:
        print("No real images found. Creating samples...")
        os.makedirs(real_dir, exist_ok=True)
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(f'{real_dir}/real_sample_{i+1}.jpg', img)
        real_paths = glob.glob(os.path.join(real_dir, '*.jpg'))
    
    if not fake_paths:
        print("No fake images found. Creating samples...")
        os.makedirs(fake_dir, exist_ok=True)
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = cv2.resize(img, (50, 50))
            img = cv2.resize(img, (100, 100))
            cv2.imwrite(f'{fake_dir}/fake_sample_{i+1}.jpg', img)
        fake_paths = glob.glob(os.path.join(fake_dir, '*.jpg'))
    
    real_train, real_temp = train_test_split(real_paths, test_size=test_size*2, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
    
    fake_train, fake_temp = train_test_split(fake_paths, test_size=test_size*2, random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)
    
    return (real_train, fake_train), (real_val, fake_val), (real_test, fake_test)