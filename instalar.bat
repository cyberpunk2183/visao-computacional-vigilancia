@echo off
chcp 65001 >nul
set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo ================================================
echo   INSTALADOR - Agente de Vigilancia Local
echo ================================================
echo.

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python 3.12 nao encontrado.
    echo.
    echo Instale com:
    echo winget install --id Python.Python.3.12 --scope user
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.12 encontrado.
echo.
echo [1/4] Atualizando pip...
"%PYTHON_EXE%" -m pip install --no-input --disable-pip-version-check --no-cache-dir -i https://pypi.org/simple --upgrade pip

echo.
echo [2/4] Instalando NumPy...
"%PYTHON_EXE%" -m pip install --no-input --disable-pip-version-check --no-cache-dir -i https://pypi.org/simple numpy

echo.
echo [3/4] Instalando OpenCV...
"%PYTHON_EXE%" -m pip install --no-input --disable-pip-version-check --no-cache-dir -i https://pypi.org/simple opencv-python

echo.
echo [4/4] Instalando YOLOv8 e PyTorch...
"%PYTHON_EXE%" -m pip install --no-input --disable-pip-version-check --no-cache-dir -i https://pypi.org/simple ultralytics

echo.
echo ================================================
echo   Instalacao concluida.
echo   Execute o arquivo: executar.bat
echo ================================================
pause
