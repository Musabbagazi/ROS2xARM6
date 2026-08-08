@echo off
title xArm 6 - Gripper Reset / Recovery
echo ============================================
echo    xArm 6  -  gripper reset
echo ============================================
echo.
echo  Use this when a run stopped with the
echo  gripper still CLOSED (possibly holding a
echo  cube), or when it reports that the gripper
echo  "did not open".
echo.
echo  It reads the fault first, then - only after
echo  you confirm - clears it, re-enables the
echo  gripper and OPENS the fingers.
echo.
echo  The arm itself is NOT moved.
echo  Hold any clamped cube: it will drop.
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python gripper_reset.py %*
echo.
echo --------------- session ended --------------
echo.
pause
