@echo off
title xArm 6 - Rotate Gripper
echo ============================================
echo    xArm 6  -  Rotate the gripper only
echo ============================================
echo.
echo  Only joint 6 (the gripper) turns - the
echo  rest of the arm does not move.
echo.
echo  Type  +10  or  -5  to rotate by degrees.
echo  Type  q  when done.
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
wsl -d Ubuntu-24.04 -- bash -lc "python3 ~/rotate_gripper.py"
echo.
echo --------------- session ended --------------
echo.
pause
