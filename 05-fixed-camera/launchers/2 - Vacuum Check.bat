@echo off
title xArm 6 - Fixed Camera: vacuum check (no motion)
echo ==================================================
echo    FIXED CAMERA  -  vacuum cup check
echo ==================================================
echo.
echo  THE ARM DOES NOT MOVE. The cup switches on and
echo  off while this runs.
echo.
echo  Two things it answers:
echo.
echo    1. WHICH WIRING. The cup can be plugged in
echo       (tool GPIO 0/1) or contact-connected (3/4).
echo       Get it wrong and every command is accepted,
echo       no error is raised, and nothing ever
echo       switches. This tries both and reports which
echo       one actually responds.
echo.
echo    2. DOES THE SEAL WORK. The cup goes on and
echo       STAYS ON until you press a key, so you can
echo       try a cube, a corner, a rough face - and
echo       watch the reading change each time.
echo.
echo  NOTE: the wiring probe pulses the cup on for
echo  about a second on each wiring FIRST. That short
echo  pulse is not the seal test - the long hold comes
echo  after it.
echo.
echo  Run this once when the cup is first fitted, and
echo  any time a pick reports "no seal" repeatedly.
echo.
echo  Add --hold to skip the probe and go straight to
echo  holding the cup on.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python fixed_vacuum.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
