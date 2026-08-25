@echo off
title xArm 6 - Auto Calibrate Camera
echo ============================================
echo    xArm 6  -  AUTO camera calibration
echo ============================================
echo.
echo  Needs cube_model.pt (run "1 - Capture
echo  Dataset" + "2 - Train Cube Model" once).
echo.
echo  Put ONE cube (ANY color) near the middle of
echo  the floor. It must NOT move during this.
echo.
echo  Phase A is automatic: two small moves
echo  measure the camera map, then the arm
echo  centers itself over the cube.
echo.
echo  Phase B happens ONE TIME EVER: drive the
echo  open fingers around the cube at mid-height
echo  with W/A/S/D/Q/E keys, then press ENTER.
echo  (Skipped automatically on later runs.)
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python auto_calibrate3.py %*
echo.
echo --------------- session ended --------------
echo.
pause
