@echo off
title xArm 6 - Teach Poses
echo ============================================
echo    xArm 6  -  Teach PICK / PLACE poses
echo ============================================
echo.
echo  Keep the E-STOP in hand.
echo  When hand-guide mode turns on the arm goes
echo  SOFT - hold it so nothing swings or drops.
echo.
echo  Push arm to a spot, press ENTER to record.
echo  Type  q  then ENTER when done.
echo.
echo --------------------------------------------
echo.
wsl -d Ubuntu-24.04 -- bash -lc "python3 ~/xarm6_teach.py"
echo.
echo --------------- session ended --------------
echo  Copy the pose_1 / pose_2 lines above into
echo  pick_and_place.py  (PICK and PLACE).
echo.
pause
