@echo off
cls
echo ================================================================================
echo               Crystallization Component Predictor
echo ================================================================================
echo.
echo Starting Streamlit application...
echo.
cd /d "%~dp0\05_app"
C:\Users\dassu\Miniconda3\envs\btp_env\Scripts\streamlit.exe run app.py
echo.
echo ================================================================================
pause

