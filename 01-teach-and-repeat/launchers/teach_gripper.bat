@echo off
title xArm 6 - Teach Gripper Tightness
echo ============================================
echo    xArm 6  -  Teach GRIP tightness
echo ============================================
echo.
echo  Put your box between the gripper fingers.
echo  Type  m  to auto-measure it, or type a
echo  number 0..850 to try a grip by hand.
echo    0   = fully closed (tightest)
echo    850 = fully open
echo  Type  q  when done - it prints GRIP_CLOSED.
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
wsl -d Ubuntu-24.04 -- bash -lc "python3 ~/gripper_teach.py"
echo.
echo --------------- session ended --------------
echo  Copy the GRIP_CLOSED value above into
echo  pick_and_place.py.
echo.
pause
