@echo off
title xArm 6 - Fixed Camera: calibrate (the arm MOVES)
echo ==================================================
echo    FIXED CAMERA  -  hand-eye calibration
echo ==================================================
echo.
echo  Teaches the arm where the camera is looking.
echo  Everything else in this project depends on it.
echo.
echo  YOU NEED A CALIBRATION PLATE:
echo    a flat, rigid, matt RED square, 100-150mm
echo    across. Card glued to thin ply or plastic is
echo    ideal. It must not sag when held by its middle.
echo.
echo    Why a plate and not a cube: the cup covers the
echo    top of whatever it holds, and the top is the
echo    one surface that matters. On a wide plate the
echo    top face stays visible all round the cup.
echo.
echo  BEFORE YOU START:
echo    - the camera is on its FINAL, RIGID mount.
echo      If it moves later, this is void and the
echo      arm will be aimed wrong in silence.
echo    - the camera and its stand are OUTSIDE the
echo      arm's reach.
echo    - NO other red object is in the camera's view.
echo    - the floor is clear.
echo    - THE ARM IS ALREADY JOGGED so the plate would
echo      sit about 20mm above the floor. That height
echo      becomes the lowest calibration level, which
echo      is what puts the measurements where the picks
echo      actually happen. Use mouse_jog.bat.
echo.
echo  The arm visits up to 27 poses at 40 deg/s.
echo  KEEP THE E-STOP IN HAND.
echo --------------------------------------------------
echo.
cd /d "%~dp0.."
python fixed_calibrate.py %*
echo.
echo ------------------ session ended -----------------
echo.
pause
