@echo off
echo ==================================================
echo          DeepGuard Project Setup
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version

REM Create project directory structure
echo.
echo Creating project structure...
mkdir DeepGuard 2>nul
cd DeepGuard
mkdir data 2>nul
mkdir data\real_images 2>nul
mkdir data\fake_images 2>nul
mkdir models 2>nul
mkdir src 2>nul
mkdir notebooks 2>nul

echo ✓ Project structure created

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv deepguard_env

echo ✓ Virtual environment created

REM Install packages
echo.
echo Installing required packages...
call deepguard_env\Scripts\activate.bat

echo Installing PyTorch and dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy scikit-learn matplotlib pillow tqdm albumentations gradio pyyaml tensorboard

echo ✓ Packages installed

REM Create configuration files
echo.
echo Creating configuration files...

REM Create requirements.txt
echo torch==2.0.1 > requirements.txt
echo torchvision==0.15.2 >> requirements.txt
echo opencv-python==4.8.0.74 >> requirements.txt
echo numpy==1.24.3 >> requirements.txt
echo scikit-learn==1.3.0 >> requirements.txt
echo matplotlib==3.7.1 >> requirements.txt
echo pillow==10.0.0 >> requirements.txt
echo tqdm==4.65.0 >> requirements.txt
echo albumentations==1.3.1 >> requirements.txt
echo gradio==3.44.0 >> requirements.txt
echo pyyaml==6.0.1 >> requirements.txt
echo streamlit==1.24.0 >> requirements.txt

REM Create config.yaml
echo data: > config.yaml
echo   real_dir: "data/real_images" >> config.yaml
echo   fake_dir: "data/fake_images" >> config.yaml
echo   test_size: 0.2 >> config.yaml
echo   batch_size: 16 >> config.yaml
echo. >> config.yaml
echo model: >> config.yaml
echo   num_classes: 2 >> config.yaml
echo   dropout_rate: 0.3 >> config.yaml
echo   backbone: "efficientnet_b0" >> config.yaml
echo. >> config.yaml
echo training: >> config.yaml
echo   num_epochs: 20 >> config.yaml
echo   learning_rate: 0.0001 >> config.yaml
echo   weight_decay: 0.0001 >> config.yaml
echo   patience: 5 >> config.yaml
echo. >> config.yaml
echo inference: >> config.yaml
echo   confidence_threshold: 0.75 >> config.yaml
echo   model_path: "models/best_model.pth" >> config.yaml

echo ✓ Configuration files created

REM Create sample images for testing
echo.
echo Creating sample test images...
python -c "
import cv2, numpy as np, os
os.makedirs('data/real_images', exist_ok=True)
os.makedirs('data/fake_images', exist_ok=True)

# Create sample real images (more natural patterns)
for i in range(3):
    img = np.random.rand(100, 100, 3).astype(np.float32)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f'data/real_images/real_sample_{i+1}.jpg', img)

# Create sample fake images (more artificial patterns)
for i in range(3):
    img = np.random.rand(100, 100, 3).astype(np.float32)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (100, 100))
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f'data/fake_images/fake_sample_{i+1}.jpg', img)
print('Sample images created in data/ folders')
"

echo ✓ Sample images created

REM Create main Python scripts
echo.
echo Creating main Python scripts...

REM Create train_model.py
echo import argparse > train_model.py
echo import torch >> train_model.py
echo from src import prepare_data, DeepfakeDetector, DeepfakeTrainer >> train_model.py
echo from src.utils import load_config >> train_model.py
echo import glob >> train_model.py
echo import os >> train_model.py
echo. >> train_model.py
echo def main(): >> train_model.py
echo     parser = argparse.ArgumentParser(description='Train Deepfake Detection Model') >> train_model.py
echo     parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file') >> train_model.py
echo     parser.add_argument('--epochs', type=int, help='Number of training epochs') >> train_model.py
echo     parser.add_argument('--batch_size', type=int, help='Batch size') >> train_model.py
echo     args = parser.parse_args() >> train_model.py
echo. >> train_model.py
echo     # Load configuration >> train_model.py
echo     config = load_config(args.config) >> train_model.py
echo. >> train_model.py
echo     # Set device >> train_model.py
echo     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') >> train_model.py
echo     print(f'Using device: {device}') >> train_model.py
echo. >> train_model.py
echo if __name__ == '__main__': >> train_model.py
echo     main() >> train_model.py

REM Create detect.py
echo import argparse > detect.py
echo import cv2 >> detect.py
echo from src.inference import DeepfakeDetectorInference >> detect.py
echo from src.utils import load_config >> detect.py
echo. >> detect.py
echo def main(): >> detect.py
echo     parser = argparse.ArgumentParser(description='Detect Deepfake Images') >> detect.py
echo     parser.add_argument('--image', type=str, required=True, help='Path to image file') >> detect.py
echo     parser.add_argument('--model', type=str, help='Path to model file') >> detect.py
echo     parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file') >> detect.py
echo     args = parser.parse_args() >> detect.py
echo. >> detect.py
echo     # Load configuration >> detect.py
echo     config = load_config(args.config) >> detect.py
echo     model_path = args.model or config['inference']['model_path'] >> detect.py
echo. >> detect.py
echo     # Initialize detector >> detect.py
echo     detector = DeepfakeDetectorInference(model_path) >> detect.py
echo. >> detect.py
echo if __name__ == '__main__': >> detect.py
echo     main() >> detect.py

echo ✓ Main scripts created

REM Create __init__.py files
echo. > src\__init__.py
echo. > notebooks\__init__.py

echo.
echo ==================================================
echo          SETUP COMPLETED SUCCESSFULLY!
echo ==================================================
echo.
echo Next steps:
echo 1. Run activate_deepguard.bat to activate environment
echo 2. Add your real images to data\real_images\
echo 3. Add your fake images to data\fake_images\ 
echo 4. Run train_model.py to train the model
echo 5. Run detect.py --image path\to\image.jpg to test
echo.
pause