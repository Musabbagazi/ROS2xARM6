@echo off
title xArm 6 - Recover / Clear Fault
echo ============================================
echo    xArm 6  -  Recover after a stop
echo ============================================
echo.
echo  This clears the error, re-enables the arm,
echo  and can OPEN the gripper to release a box
echo  stuck in mid-air.
echo.
echo  HOLD THE BOX before you let it open!
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
wsl -d Ubuntu-24.04 -- bash -lc "python3 ~/recover.py"
echo.
echo --------------- session ended --------------
echo.
pause
