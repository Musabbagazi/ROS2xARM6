@echo off
title xArm 6 - Train Cube YOLO Model
echo ============================================
echo    xArm 6  -  train the cube model
echo ============================================
echo.
echo  Trains YOLOv8n on the captured dataset
echo  (GPU, ~10-20 min). The arm is NOT used.
echo.
echo  Warm-starts from the existing cube_model.pt
echo  and backs it up to cube_model_prev.pt, so
echo  new scenes are ADDED without forgetting the
echo  old ones.
echo    python train_cubes.py [epochs] [fresh]
echo  Pass "fresh" to train from scratch instead.
echo.
echo  Output: cube_model.pt
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python train_cubes.py %*
echo.
echo --------------- session ended --------------
echo.
pause
