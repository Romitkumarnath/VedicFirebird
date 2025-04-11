@echo off
echo Starting Vedic Astrology Bot...
python run_astrology_app.py
if %ERRORLEVEL% NEQ 0 (
    echo An error occurred while running the application.
    echo Press any key to exit...
    pause > nul
) 