#!/usr/bin/env python3
"""
Test multiple images quickly
"""

import os
import glob
import subprocess

# Find some fake images
fake_images = glob.glob('data/fake_images/*.jpg')[:5]  # Test first 5

print("Testing fake images...")
for img_path in fake_images:
    print(f"\nTesting: {os.path.basename(img_path)}")
    result = subprocess.run(['python', 'detect.py', '--image', img_path], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)