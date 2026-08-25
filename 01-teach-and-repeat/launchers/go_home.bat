@echo off
title xArm 6 - Go Home
echo ============================================
echo    xArm 6  -  Go to HOME position
echo ============================================
echo.
echo  Moves the arm to the saved HOME
echo  (gripper straight, joint6 = 4.25 deg).
echo.
echo  Keep the E-STOP in hand.
echo  You will be asked  y/N  before it moves.
echo --------------------------------------------
echo.
wsl -d Ubuntu-24.04 -- bash -lc "python3 ~/xarm6_go_home.py"
echo.
echo --------------- session ended --------------
echo.
pause
