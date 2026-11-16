@echo off
set YOLO_CONFIG_DIR=%~dp0vigilancia\ultralytics
set MPLCONFIGDIR=%~dp0vigilancia\matplotlib
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" %*
