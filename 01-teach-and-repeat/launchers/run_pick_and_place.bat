@echo off
title xArm 6 - Pick and Place
echo ============================================
echo    xArm 6  -  Pick and Place
echo ============================================
echo.
echo  Sequence:
echo    HOME  -^>  open gripper
echo    move above PICK  -^>  lower  -^>  GRIP
echo    lift  -^>  move above DROP  -^>  lower
echo    RELEASE  -^>  retreat  -^>  HOME
echo.
echo  Keep the E-STOP in hand. Runs slowly.
echo  You will be asked  y/N  before it moves.
echo.
echo --------------------------------------------
echo.
wsl -d Ubuntu-24.04 -- bash -lc "python3 ~/pick_and_place.py"
echo.
echo --------------- session ended --------------
echo.
pause
