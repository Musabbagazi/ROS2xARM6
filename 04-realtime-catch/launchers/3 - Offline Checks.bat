@echo off
title xArm 6 - Catch offline checks
echo ============================================
echo    Real-time catch  -  OFFLINE CHECKS
echo ============================================
echo.
echo  Arithmetic only. No arm, no camera, no
echo  model - safe to run any time, anywhere.
echo.
echo  Checks the velocity fit, the gates that
echo  refuse an uncatchable motion, the intercept
echo  planner, and that the aim math still agrees
echo  exactly with the v3 mapping.
echo --------------------------------------------
echo.
cd /d "%~dp0.."
python test_catch.py %*
echo.
echo --------------------------------------------
echo.
pause
