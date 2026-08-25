@echo off
title xArm 6 - Mouse Jog
echo ============================================
echo    xArm 6  -  MOUSE jog  (X/Y by mouse)
echo ============================================
echo.
echo   HOLD LEFT MOUSE BUTTON = arm follows mouse
echo   release it and the arm stops.
echo.
echo   mouse      = X / Y        w / s = Z up / down
echo   q / e      = rotate yaw   o / p = gripper open / close
echo   1..5       = 0.05 / 0.1 / 0.25 / 0.5 / 1.0 mm per pixel
echo   SPACE      = record pose  ESC   = quit
echo.
echo   Keep the E-STOP in hand. Start slow (key 1 or 2).
echo --------------------------------------------
echo.
pause
cd /d "G:\ROS 2\vision"
python mouse_jog.py
echo.
echo --------------- session ended --------------
echo.
pause
