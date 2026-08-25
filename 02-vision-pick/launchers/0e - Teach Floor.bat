@echo off
title xArm 6 - Teach Floor (for a see-through floor)
echo ============================================
echo    xArm 6  -  teach the floor plane
echo ============================================
echo.
echo  ONLY needed for a floor the depth camera
echo  cannot see - glass or clear acrylic. The
echo  infrared goes straight through it, so the
echo  floor is measured ONCE through an opaque
echo  sheet and stored.
echo.
echo  On a normal opaque floor, skip this and
echo  delete floor_ref.json - the floor is then
echo  fitted live on every frame.
echo.
echo  BEFORE you start:
echo    1. Cover the floor with a flat opaque
echo       sheet - paper, card, a thin mat.
echo    2. Take the CUBES OFF the floor.
echo.
echo  The arm moves to the SCAN pose. Keep the
echo  E-STOP in hand.
echo.
echo  AFTER it finishes: take the sheet away.
echo.
echo  Re-run this if the floor, the camera or
echo  the SCAN pose ever changes.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python teach_floor.py %*
echo.
echo --------------- session ended --------------
echo.
pause
