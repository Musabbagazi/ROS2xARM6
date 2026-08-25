@echo off
title xArm 6 - Fixed Camera: offline checks
echo ==================================================
echo    FIXED CAMERA  -  offline checks
echo ==================================================
echo.
echo  No arm, no camera, no model. Arithmetic only,
echo  safe anywhere, any time.
echo.
echo  Renders a synthetic cell - a floor, a cube, a
echo  camera 45 degrees off to the side - and pushes
echo  it through the real detector and the real
echo  calibration, so a sign error in the transform
echo  is caught here rather than by an arm reaching
echo  for the mirror image of a cube.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python test_fixed.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
