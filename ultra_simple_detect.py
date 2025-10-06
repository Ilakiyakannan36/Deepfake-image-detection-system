#!/usr/bin/env python3
"""
ULTRA SIMPLE Deepfake Detection
"""

import cv2
import numpy as np
import joblib
import argparse

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (64, 64))
        features = []
        
        # Same features as training
        for channel in range(3):
            features.append(np.mean(img[:, :, channel]))
            features.append(np.std(img[:, :, channel]))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        return np.array(features).reshape(1, -1)
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Simple Deepfake Detection')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', default='models/simple_rf_model.pkl', help='Path to model file')
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = joblib.load(args.model)
    except:
        print("❌ Model not found. Please train first with: python ultra_simple_train.py")
        return
    
    # Extract features
    features = extract_features(args.image)
    if features is None:
        print(f"❌ Could not process image: {args.image}")
        return
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    result = {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': max(probability),
        'real_probability': probability[0],
        'fake_probability': probability[1]
    }
    
    print("\n" + "="*50)
    print("DEEPGUARD DETECTION RESULTS")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Real Probability: {result['real_probability']:.4f}")
    print(f"Fake Probability: {result['fake_probability']:.4f}")
    
    if result['prediction'] == 'FAKE' and result['confidence'] > 0.7:
        print("⚠️  WARNING: This image is likely manipulated!")
    elif result['prediction'] == 'REAL' and result['confidence'] > 0.7:
        print("✅ This image appears to be authentic!")
    else:
        print("❓ Uncertain prediction")

if __name__ == "__main__":
    main()