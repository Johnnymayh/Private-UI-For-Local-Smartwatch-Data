@echo off
REM Health Dashboard launcher.
REM Keep this file in the SAME folder as app.py and your Gadgetbridge file.
REM Double-click it any time to start the dashboard - no need to open a
REM terminal or navigate to the folder yourself.

cd /d "%~dp0"

echo Starting Personal Health Dashboard...
echo (Your browser should open automatically. Leave this window open while using it.)
echo.

python -m streamlit run app.py

echo.
echo Dashboard stopped. Press any key to close this window.
pause >nul
