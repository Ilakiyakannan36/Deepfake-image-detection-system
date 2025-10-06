#!/usr/bin/env python3
"""
DeepGuard Detection Script - FIXED VERSION
"""

import cv2
import numpy as np
import joblib
import argparse
import os
import sys

def extract_advanced_features(image_path):
    """Extract the same features used during training"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not read image: {image_path}")
            return None
        
        img = cv2.resize(img, (128, 128))
        features = []
        
        # Same features as training
        for channel in range(3):
            features.append(np.mean(img[:, :, channel]))
            features.append(np.std(img[:, :, channel]))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.max(gray))
        features.append(np.min(gray))
        
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.mean(edges))
        features.append(np.std(edges))
        features.append(np.sum(edges > 0) / edges.size)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.var(laplacian))
        features.append(np.median(gray))
        
        return np.array(features)
        
    except Exception as e:
        print(f"ERROR: Error processing image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Deepfake Image Detection')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', default='models/deepguard_advanced_model.pkl', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print("ERROR: Model not found. Please train first with: python train_model_final.py")
        return
    
    # Load model
    try:
        model = joblib.load(args.model)
        print("SUCCESS: Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        return
    
    # Extract features and predict
    features = extract_advanced_features(args.image)
    if features is None:
        return
    
    prediction = model.predict(features.reshape(1, -1))[0]
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    
    # Display results
    print("\n" + "="*60)
    print("          DEEPGUARD DETECTION RESULTS")
    print("="*60)
    print(f"Image: {os.path.basename(args.image)}")
    print(f"Prediction: {'REAL' if prediction == 0 else 'FAKE'}")
    print(f"Confidence: {max(probabilities):.3f}")
    print(f"Real Probability: {probabilities[0]:.3f}")
    print(f"Fake Probability: {probabilities[1]:.3f}")
    print("="*60)
    
    # Interpretation
    if prediction == 1 and probabilities[1] > 0.7:
        print("WARNING: This image is likely manipulated or AI-generated!")
    elif prediction == 0 and probabilities[0] > 0.7:
        print("SUCCESS: This image appears to be authentic!")
    elif max(probabilities) > 0.6:
        status = 'REAL' if prediction == 0 else 'FAKE'
        print(f"LIKELY {status} (moderate confidence)")
    else:
        print("UNCERTAIN: Low confidence prediction")
    
    print("="*60)

if __name__ == "__main__":
    main()