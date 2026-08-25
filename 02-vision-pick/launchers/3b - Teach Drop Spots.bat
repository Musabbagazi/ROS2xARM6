@echo off
title xArm 6 - Teach Drop Spots (colour sort)
echo ============================================
echo    xArm 6  -  teach the drop spots
echo ============================================
echo.
echo  Sets where each colour is released. BLUE is
echo  required for colour sorting; RED defaults to
echo  the existing PLACE.
echo.
echo  The arm goes above the current red spot, then
echo  you WASD-jog the gripper to the drop spot and
echo  press ENTER:
echo    w/s = +/-X   a/d = +/-Y   q/e = Z up/down
echo    1-5 = step size            ENTER = accept
echo.
echo  Keep both spots reasonably central - a spot
echo  far out on one side makes the arm refuse
echo  cubes on the other side under the joint
echo  sweep guard.
echo.
echo  Keep the E-STOP in hand.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python teach_place.py %*
echo.
echo --------------- session ended --------------
echo.
pause
