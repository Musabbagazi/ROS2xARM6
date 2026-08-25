@echo off
title xArm 6 - Catch View (camera only, nothing moves)
echo ============================================
echo    Real-time catch  -  TRACKER VIEW
echo ============================================
echo.
echo  Camera only. The arm is NOT commanded and
echo  does not need to be powered.
echo.
echo  Slide a cube across the view and watch:
echo    - the fitted velocity arrow
echo    - speed, heading and fit residual
echo    - CATCHABLE, or the exact refusal
echo.
echo  Run this BEFORE the catch itself, and again
echo  any time the cell or the lighting changes.
echo  A cube that does not read CATCHABLE here
echo  will be refused there too - this way you
echo  find out without an arm in the room.
echo.
echo  Add the robot IP to read the arm's real pose
echo  (still read-only, it is never commanded):
echo      "1 - Catch View (no arm).bat" --ip 192.168.1.197
echo.
echo  Needs calib3.json + grip_ref.json in vision\
echo  (run the v3 "3 - Auto Calibrate" first).
echo.
echo  ESC or q in the image window to quit.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python catch_view.py %*
echo.
echo --------------- session ended --------------
echo.
pause
