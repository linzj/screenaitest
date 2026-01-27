@echo off
REM Build script for chrome_ocr_rs
REM One-click build with prebuilt TFLite library

setlocal

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

set TFLITEC_PREBUILT_PATH=%SCRIPT_DIR%\libs\tensorflowlite_c.dll
set TFLITEC_HEADER_DIR=%SCRIPT_DIR%\include

cd /d "%SCRIPT_DIR%"
cargo build --release
if errorlevel 1 goto :error

REM Copy runtime DLLs to output directory
if exist target\release\chrome_ocr.exe (
    copy /Y libs\tensorflowlite_c.dll target\release\
    echo Build complete: target\release\chrome_ocr.exe
)
goto :end

:error
echo Build failed
exit /b 1

:end
