#!/usr/bin/env python3
"""
ULTRA SIMPLE Deepfake Training - No torchvision required!
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import glob

print("🚀 Starting Ultra Simple Deepfake Training...")

# Create data directories if they don't exist
os.makedirs('data/real_images', exist_ok=True)
os.makedirs('data/fake_images', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Find images
real_paths = glob.glob('data/real_images/*.jpg') + glob.glob('data/real_images/*.png')
fake_paths = glob.glob('data/fake_images/*.jpg') + glob.glob('data/fake_images/*.png')

# Create sample images if none found
if not real_paths or not fake_paths:
    print("Creating sample images...")
    for i in range(20):
        # Real images (more natural)
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(f'data/real_images/real_{i}.jpg', img)
        
        # Fake images (more artificial patterns)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = cv2.blur(img, (3, 3))
        cv2.imwrite(f'data/fake_images/fake_{i}.jpg', img)
    
    real_paths = glob.glob('data/real_images/*.jpg')
    fake_paths = glob.glob('data/fake_images/*.jpg')

print(f"Found {len(real_paths)} real images and {len(fake_paths)} fake images")

# Extract simple features from images
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(10)  # Return zeros if image can't be loaded
        
        img = cv2.resize(img, (64, 64))  # Resize to smaller size
        features = []
        
        # Simple features: mean and std of each channel
        for channel in range(3):
            features.append(np.mean(img[:, :, channel]))
            features.append(np.std(img[:, :, channel]))
        
        # Add some basic texture features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        return np.array(features)
    except:
        return np.zeros(10)

print("Extracting features from images...")

# Extract features from all images
X = []
y = []

for path in real_paths:
    features = extract_features(path)
    X.append(features)
    y.append(0)  # 0 for real

for path in fake_paths:
    features = extract_features(path)
    X.append(features)
    y.append(1)  # 1 for fake

X = np.array(X)
y = np.array(y)

print(f"Feature matrix shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train a simple Random Forest classifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Training completed!")
print(f"📊 Test Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'models/simple_rf_model.pkl')
print("💾 Model saved as 'models/simple_rf_model.pkl'")

# Save results
results = {
    'accuracy': float(accuracy),
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'real_images': int(len(real_paths)),
    'fake_images': int(len(fake_paths))
}

with open('training_results.json', 'w') as f:
    import json
    json.dump(results, f, indent=4)

print("📊 Results saved to 'training_results.json'")
print("\n🎉 Ready to detect images! Run: python ultra_simple_detect.py --image your_image.jpg")