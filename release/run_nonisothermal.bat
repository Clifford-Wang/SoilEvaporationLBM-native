@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "SoilEvaporationLBM_NonIsothermal.exe" (
    echo ERROR: SoilEvaporationLBM_NonIsothermal.exe was not found in this folder.
    pause
    exit /b 1
)
if not exist "config_nonisothermal.txt" (
    echo ERROR: config_nonisothermal.txt was not found in this folder.
    pause
    exit /b 1
)
"SoilEvaporationLBM_NonIsothermal.exe" "config_nonisothermal.txt"
set "exit_code=%errorlevel%"
if not "%exit_code%"=="0" (
    echo Non-isothermal simulation failed with exit code %exit_code%.
    pause
)
exit /b %exit_code%
