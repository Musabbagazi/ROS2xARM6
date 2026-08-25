@echo off
setlocal
title xArm 6 - RESET EVERYTHING
cd /d "%~dp0"
echo ============================================
echo    xArm 6  -  RESET EVERYTHING
echo ============================================
echo.
echo  Run this when a run stopped badly and the
echo  other .bat files will not move the arm any
echo  more - typically stopping with
echo    "arm still in error [13, 0]".
echo.
echo  It will:
echo    1. close leftover script windows that are
echo       still holding the arm and the camera
echo    2. restart WSL (Ubuntu) if you want it
echo    3. re-plug the RealSense camera in software
echo    4. ping the controller
echo    5. clear the fault, re-enable the arm and
echo       re-arm the gripper
echo    6. if it will not clear, say exactly why
echo       and wait while you power-cycle
echo.
echo  The ARM IS NEVER MOVED by this.
echo  The GRIPPER CAN OPEN - you are asked first,
echo  so HOLD any cube that is clamped in it.
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
pause

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Python is not on PATH in this window, so nothing
    echo  here can run. Open a new terminal, or reinstall
    echo  Python 3.12 with "Add to PATH" ticked.
    echo.
    pause
    exit /b 1
)

echo.
echo --- 0) WSL (Ubuntu) -------------------------
where wsl >nul 2>&1
if errorlevel 1 (
    echo    wsl is not installed - nothing to restart.
    goto :afterwsl
)
echo    go_home.bat, recover.bat and run_pick_and_place.bat
echo    run their Python inside Ubuntu-24.04. Restarting WSL
echo    kills anything hung in there that is still holding
echo    the arm's connection. It costs a few seconds on the
echo    next WSL launcher and nothing else.
echo.
choice /c YN /n /m "   Restart WSL now? [Y/N] "
if errorlevel 2 (
    echo    left running.
    goto :afterwsl
)
echo    shutting WSL down ...
wsl --shutdown
echo    done - it will start again by itself next time.
:afterwsl

python "%~dp0reset_all.py" %*
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
    echo --------------- reset complete -------------
) else (
    echo ------------- STILL NOT READY --------------
    echo  Do the step printed above, then run this
    echo  file again.
)
echo.
pause
endlocal
