@echo off
title xArm 6 - Fixed Camera: pick DRY RUN (no motion)
echo ==================================================
echo    FIXED CAMERA  -  pick, DRY RUN
echo ==================================================
echo.
echo  THE ARM DOES NOT MOVE.
echo.
echo  It connects only to ask the controller which
echo  poses are reachable, then walks the whole floor
echo  and prints, for every cube it can see:
echo.
echo    - the exact approach and grasp poses it would
echo      command,
echo    - the gripper openings it would use,
echo    - or the reason it would refuse that cube.
echo.
echo  Run this after every calibration, and any time
echo  the cell changes. It is the cheapest way to find
echo  out that the transform is wrong.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python fixed_pick.py --dry %*
echo.
echo ------------------ session ended -----------------
echo.
pause
