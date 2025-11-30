@echo off
REM Quick activation script for rahatai environment
REM Use this if conda activate doesn't work

REM Try standard conda activate
call conda activate rahatai 2>nul

if %ERRORLEVEL% EQU 0 (
    echo Environment activated successfully!
    python --version
    goto :end
)

REM Try alternative method using conda's activate script
if defined CONDA_PREFIX (
    call "%CONDA_PREFIX%\Scripts\activate.bat" rahatai 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo Environment activated successfully!
        python --version
        goto :end
    )
)

REM Try finding conda installation
for %%i in (conda.exe) do set CONDA_PATH=%%~$PATH:i
if defined CONDA_PATH (
    for /f "delims=" %%i in ("%CONDA_PATH%") do set CONDA_DIR=%%~dpi
    if defined CONDA_DIR (
        set "CONDA_ENV_PATH=%CONDA_DIR%..\envs\rahatai"
        if exist "%CONDA_ENV_PATH%\Scripts\activate.bat" (
            call "%CONDA_ENV_PATH%\Scripts\activate.bat"
            echo Environment activated successfully!
            python --version
            goto :end
        )
    )
)

echo.
echo ============================================================
echo Could not activate conda environment automatically
echo ============================================================
echo.
echo Please try one of these methods:
echo.
echo Method 1: Restart terminal after conda init
echo   1. Close this terminal
echo   2. Open a NEW terminal
echo   3. Run: conda activate rahatai
echo.
echo Method 2: Use Anaconda Prompt
echo   1. Open "Anaconda Prompt" from Start Menu
echo   2. Navigate to: cd F:\RAHATAIusman\RAHATAI
echo   3. Run: conda activate rahatai
echo.
echo Method 3: Use PowerShell
echo   1. Open PowerShell
echo   2. Run: conda init powershell
echo   3. Restart PowerShell
echo   4. Run: conda activate rahatai
echo.
pause

:end

