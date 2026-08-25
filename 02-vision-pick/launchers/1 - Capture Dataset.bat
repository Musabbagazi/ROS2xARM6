@echo off
title xArm 6 - Capture YOLO Cube Dataset
echo ============================================
echo    xArm 6  -  cube dataset capture
echo ============================================
echo.
echo  Scatter 2-5 cubes on the floor - use EVERY
echo  cube color you own. The arm hovers at a few
echo  heights and snaps frames; boxes are drawn
echo  automatically from depth (no hand labeling).
echo.
echo  Between rounds: rearrange / swap the cubes,
echo  keep them SEPARATED (a finger apart, never
echo  touching or stacked), then take your hands
echo  OUT of view and press ENTER.
echo.
echo  Appends to an existing dataset - it does not
echo  overwrite one.
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python capture_dataset.py %*
echo.
echo --------------- session ended --------------
echo.
pause
