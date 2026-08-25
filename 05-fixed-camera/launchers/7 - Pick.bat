@echo off
title xArm 6 - Fixed Camera: PICK (the arm MOVES)
echo ==================================================
echo    FIXED CAMERA  -  pick
echo ==================================================
echo.
echo  THE ARM MOVES at 40 deg/s.
echo  KEEP THE E-STOP IN HAND.
echo.
echo  One fixed camera sees the whole cell at once, so
echo  there is no scan pose and no search pattern. The
echo  arm is told where every cube is and goes.
echo.
echo  Red cubes first, then blue, then anything the
echo  classifier could not call.
echo.
echo  Before the first move it checks whether the
echo  camera has been knocked since it was calibrated,
echo  and stops if it has. That check is the whole
echo  reason a fixed camera is safe to trust.
echo.
echo  Run "4 - Pick DRY RUN" first if anything about
echo  the cell has changed.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python fixed_pick.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
