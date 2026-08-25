@echo off
title xArm 6 - Real-Time Catch (the arm MOVES)
echo ============================================
echo    xArm 6  -  catching a MOVING cube
echo ============================================
echo.
echo  The arm watches a cube move, works out where
echo  it is going, goes to a spot AHEAD of it, and
echo  closes as it arrives. It never waits for the
echo  cube to stop.
echo.
echo  Send the cube:
echo    - in a straight line at a steady speed
echo    - roughly 2 to 17 cm/s
echo    - ACROSS the cell, not away from the arm
echo    - flat, not tumbling
echo    - LET GO of it. A cube still in your hand
echo      is the handoff project's job.
echo.
echo  Between committing and closing the arm SITS
echo  STILL in the cube's path with the fingers
echo  open. That is the design - the camera cannot
echo  see that close, so the catch is timed.
echo.
echo  Run "1 - Catch View (no arm).bat" first if
echo  anything about the cell has changed.
echo.
echo  Needs calib3.json + grip_ref.json in vision\
echo  (run the v3 "3 - Auto Calibrate" first) and
echo  cube_model.pt (v3 "2 - Train Cube Model").
echo.
echo  ESC or q in THIS window stops it between
echo  steps. KEEP THE E-STOP IN HAND.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python catch_pick.py %*
echo.
echo --------------- session ended --------------
echo.
pause
