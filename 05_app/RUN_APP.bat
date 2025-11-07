@echo off
cls
echo ================================================================================
echo    Crystallization Component Predictor - Interactive Web App
echo ================================================================================
echo.
echo Starting Streamlit application...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Features:
echo   - Adjust sliders for crystallization parameters
echo   - Get real-time predictions
echo   - Compare Simple vs Advanced baseline
echo   - View Top-5 component predictions
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

cd /d "%~dp0"
C:\Users\dassu\Miniconda3\envs\btp_env\Scripts\streamlit.exe run app.py

pause
