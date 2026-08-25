@echo off
title xArm 6 - Sorting Cell: offline checks
echo ==================================================
echo    SORTING CELL  -  offline checks
echo ==================================================
echo.
echo  No arm, no camera, no model. Arithmetic only,
echo  safe anywhere, any time.
echo.
echo  Builds a synthetic cell - a table, a 30mm cube,
echo  a camera 45 degrees off to the side - and pushes
echo  it through the real measurement code, including
echo  the case that decides this project: a suction
echo  cup standing on the cube's top face, occluding
echo  the middle and leaving a 3mm ring.
echo.
echo  A sign error is caught here rather than by an
echo  arm reaching for the mirror image of a cube.
echo --------------------------------------------------
echo.
cd /d "G:\ROS 2\cell"
python test_cell.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
