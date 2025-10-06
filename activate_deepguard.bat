@echo off
echo ==================================================
echo       DeepGuard Environment Activator
echo ==================================================
echo.

REM Check if we're in the right directory
if not exist "deepguard_env" (
    echo ERROR: deepguard_env folder not found!
    echo Please run this script from the DeepGuard folder
    echo or run setup.bat first to create the project.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating DeepGuard environment...
call deepguard_env\Scripts\activate.bat

REM Set project directory
set PROJECT_DIR=%CD%

echo.
echo ✓ Environment activated: deepguard_env
echo ✓ Project directory: %PROJECT_DIR%
echo.

REM Display menu
:menu
echo ============ DEEPGUARD MENU ============
echo 1. Train Model
echo 2. Detect Image
echo 3. Launch Web Interface
echo 4. Install Missing Packages
echo 5. Deactivate Environment
echo 6. Exit
echo ========================================
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto detect
if "%choice%"=="3" goto web
if "%choice%"=="4" goto install
if "%choice%"=="5" goto deactivate
if "%choice%"=="6" goto exit

echo Invalid choice! Please try again.
echo.
goto menu

:train
echo.
echo Starting model training...
echo Press Ctrl+C to stop training
python train_model.py
echo.
pause
goto menu

:detect
echo.
set /p image_path="Enter image path: "
if "%image_path%"=="" (
    echo No image path provided!
    pause
    goto menu
)
python detect.py --image "%image_path%"
echo.
pause
goto menu

:web
echo.
echo Launching Web Interface...
echo Open http://localhost:7860 in your browser
echo Press Ctrl+C to stop the server
python web_interface.py
echo.
pause
goto menu

:install
echo.
echo Installing required packages...
pip install -r requirements.txt
echo.
echo ✓ Packages installed/updated
pause
goto menu

:deactivate
echo.
echo Deactivating environment...
deactivate
echo Environment deactivated
pause
goto exit

:exit
echo.
echo Thank you for using DeepGuard!
pause