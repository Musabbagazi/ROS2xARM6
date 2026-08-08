@echo off
title xArm 6 - Vision Pick (colour priority)
echo ============================================
echo    xArm 6  -  autonomous cube pick
echo ============================================
echo.
echo  Clears the floor in COLOUR ORDER:
echo    every RED cube, then every BLUE one,
echo    then anything left over.
echo.
echo  Cube size, grab height and grasp angle all
echo  come from depth - any cube, anywhere, no
echo  taught grab pose. If a cube is moving the
echo  arm waits for it to hold still, and retries
echo  if it moves mid-pick.
echo.
echo  Needs calib3.json + grip_ref.json
echo  (run "3 - Auto Calibrate" first) and
echo  places.json for the blue drop spot
echo  (run "3b - Teach Drop Spots").
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python vision_pick3.py %*
echo.
echo --------------- session ended --------------
echo.
pause
