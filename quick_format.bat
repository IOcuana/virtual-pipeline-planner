@echo off
setlocal ENABLEDELAYEDEXPANSION

REM --- Run from the folder this .bat lives in ---
cd /d "%~dp0"

REM --- Prefer project venv if present ---
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

REM --- Verify tools exist (either in venv or globally) ---
where ruff >nul 2>&1 || (
  echo [!] Ruff not found. Run format_lint.bat once to install tools.
  exit /b 1
)
where black >nul 2>&1 || (
  echo [!] Black not found. Run format_lint.bat once to install tools.
  exit /b 1
)

REM --- Parse args: --check for CI-style check (no changes), otherwise auto-fix
set MODE=fix
set TARGETS=.

if "%~1"=="" goto run
if /I "%~1"=="--check" (
  set MODE=check
  shift
)
if not "%~1"=="" (
  set TARGETS=%*
)

:run
echo [*] Targets: %TARGETS%
if /I "%MODE%"=="check" (
  echo [*] Ruff: checking...
  ruff check %TARGETS%
  set ERR1=!ERRORLEVEL!
  echo [*] Black: checking...
  black --check %TARGETS%
  set ERR2=!ERRORLEVEL!
  if NOT "!ERR1!!ERR2!"=="00" (
    echo [!] Formatting/lint issues found.
    exit /b 1
  ) else (
    echo [✓] Clean: no issues found.
    exit /b 0
  )
) else (
  echo [*] Ruff: auto-fixing...
  ruff check %TARGETS% --fix
  echo [*] Black: formatting...
  black %TARGETS%
  echo [✓] Done.
)

if "%1"=="" (
  echo.
  pause
)
endlocal
