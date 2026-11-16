@echo off
chcp 65001 >nul
set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo ================================================
echo   AGENTE DE VIGILANCIA LOCAL
echo   Iniciando monitoramento...
echo ================================================
echo.

"%PYTHON_EXE%" vigilancia.py
pause
