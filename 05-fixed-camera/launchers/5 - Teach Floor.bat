@echo off
title xArm 6 - Fixed Camera: teach the floor (no motion)
echo ==================================================
echo    FIXED CAMERA  -  teach the support surface
echo ==================================================
echo.
echo  THE ARM IS NOT MOVED BY THIS.
echo.
echo  The cell's glass floor is invisible to the depth
echo  camera - its infrared goes straight through and
echo  reports what is underneath. So the surface has
echo  to be measured once, through something opaque,
echo  and stored. Because the camera does not move,
echo  once is enough.
echo.
echo  BEFORE YOU START:
echo    - every cube OFF the floor,
echo    - flat opaque paper or card laid over the
echo      working area (partial coverage is fine),
echo    - the arm parked out of the camera's view
echo      (go_home.bat).
echo.
echo  Run "1 - Calibrate Camera" first - the surface
echo  is stored in the robot's frame, so there has to
echo  be a frame to store it in.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python fixed_floor.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
