# One-command setup for Windows (task 1.1).
# Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Creates .venv with Python 3.13, installs the core packages and the project, then runs the tests.
# Training packages (PyTorch with CUDA, ultralytics) are installed separately, see README.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Fail($message) {
    Write-Host ""
    Write-Host "SETUP STOPPED: $message" -ForegroundColor Red
    exit 1
}

Write-Host "[1/4] Checking for Python 3.13..." -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail "the 'py' launcher was not found. Install Python 3.13 from python.org and tick 'Add python.exe to PATH'."
}
$version = & py -3.13 -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
if ($LASTEXITCODE -ne 0 -or $version -ne "3.13") {
    Fail "Python 3.13 is not installed (py -3.13 did not work). Install Python 3.13, not 3.14."
}
Write-Host "      Found Python $version"

Write-Host "[2/4] Creating the virtual environment (.venv)..." -ForegroundColor Cyan
if (Test-Path ".venv\Scripts\python.exe") {
    $venvVersion = & .venv\Scripts\python.exe -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($venvVersion -ne "3.13") {
        Fail ".venv exists but uses Python $venvVersion. Delete the .venv folder and run this script again."
    }
    Write-Host "      .venv already exists, reusing it"
} else {
    & py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "could not create .venv." }
}
$python = ".venv\Scripts\python.exe"

Write-Host "[3/4] Installing packages (takes a few minutes)..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed. Check your internet connection." }
& $python -m pip install numpy opencv-python pytest --quiet
if ($LASTEXITCODE -ne 0) { Fail "installing numpy, opencv-python and pytest failed." }
& $python -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) { Fail "installing the project (pip install -e .) failed." }

Write-Host "[4/4] Running the tests..." -ForegroundColor Cyan
& $python -m pytest
if ($LASTEXITCODE -ne 0) { Fail "some tests failed. Post a screenshot of this window in the team WhatsApp group." }

Write-Host ""
Write-Host "SETUP COMPLETE. Screenshot this window and post it in the team WhatsApp group." -ForegroundColor Green
Write-Host "Next time, start work with:  .venv\Scripts\activate"
