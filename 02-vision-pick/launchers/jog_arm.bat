@echo off
title xArm 6 - WASD Drive
echo ============================================
echo    xArm 6  -  WASD keyboard drive
echo ============================================
echo.
echo  Drive the arm with single keys:
echo    w/s = X    a/d = Y    q/e = up/down
echo    z/c = rotate     o/p = gripper
echo    1..5 = step size 1/5/10/20/40 mm
echo    SPACE = record pose    ENTER = finish
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
cd /d "G:\ROS 2\vision"
python jog_arm.py
echo.
echo --------------- session ended --------------
echo.
pause
