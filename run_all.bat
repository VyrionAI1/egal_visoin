@echo off
echo ========================================================
echo        EAGLEVISION - LOCAL MICROSERVICES LAUNCHER
echo ========================================================
echo.
echo Please ensure your Kafka/Zookeeper server is already running!
echo.
echo [1/3] Starting Database Consumer (Background)...
start "EagleVision Consumer" cmd /k "python consumer.py"

echo [2/3] Starting Streamlit Dashboard (Background)...
start "EagleVision Dashboard" cmd /k "streamlit run dashboard.py"

echo.
echo [3/3] Starting Main CV Processor...
echo (Press 'q' in the video window to safely close and export video)
python main.py

echo.
echo Cleaning up background services...
taskkill /FI "WINDOWTITLE eq EagleVision Consumer*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq EagleVision Dashboard*" /T /F >nul 2>&1

echo.
echo All processes safely closed. Have a great day!
pause
