@echo off
title xArm 6 - RESET EVERYTHING
cd /d "%~dp0..\.."
python reset_all.py %*
pause
