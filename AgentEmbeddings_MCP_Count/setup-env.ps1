[CmdletBinding()]
param(
    [string]$PythonCommand = "py"
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$activatePath = Join-Path $venvPath "Scripts\Activate.ps1"
$requiredVenvFiles = @(
    $venvPython
    $activatePath
    (Join-Path $venvPath "Scripts\activate.bat")
    (Join-Path $venvPath "Scripts\activate")
)

if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
    throw "Python command '$PythonCommand' was not found. Install Python or pass -PythonCommand with a valid executable."
}

if ($requiredVenvFiles.Where({ -not (Test-Path -LiteralPath $_) }).Count -gt 0) {
    $venvExists = Test-Path -LiteralPath $venvPath
    if ($venvExists) {
        Write-Host "Removing incomplete virtual environment at '$venvPath'..."
        Remove-Item -LiteralPath $venvPath -Recurse -Force
    }

    Write-Host "Creating virtual environment at '$venvPath'..."
    & $PythonCommand -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create or repair the virtual environment."
    }

    $missingVenvFiles = $requiredVenvFiles.Where({ -not (Test-Path -LiteralPath $_) })
    if ($missingVenvFiles.Count -gt 0) {
        throw "Virtual environment is incomplete. Missing: $($missingVenvFiles -join ', ')"
    }
}
else {
    Write-Host "Using existing virtual environment at '$venvPath'."
}

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "Requirements file not found at '$requirementsPath'."
}

Write-Host "Installing project dependencies..."
& $venvPython -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install project dependencies."
}

Write-Host ""
Write-Host "Virtual environment setup completed."
Write-Host "Activate it with:"
Write-Host "  . `"$activatePath`""
