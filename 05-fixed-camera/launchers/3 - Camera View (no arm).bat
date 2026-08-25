@echo off
title xArm 6 - Fixed Camera: view (nothing moves)
echo ==================================================
echo    FIXED CAMERA  -  cube view
echo ==================================================
echo.
echo  NOTHING MOVES. No arm connection is made at all,
echo  so this is safe to run at any time, with anyone
echo  in the cell.
echo.
echo  LEFT panel  : the camera's view, with each cube
echo                outlined and each REFUSED cluster
echo                labelled with the reason.
echo  RIGHT panel : the same scene as a top-down plan
echo                in the ROBOT'S frame, in mm.
echo.
echo  The right panel is the real test of a
echo  calibration. Put a cube in a corner and check
echo  the numbers against a tape measure - a fit can
echo  have a small residual and still be wrong out
echo  where the cubes actually lie.
echo.
echo  q or ESC to quit, s to save a snapshot.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python fixed_view.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
