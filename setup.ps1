$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.11 -m venv .venv }
$requirementsFile = if (Test-Path 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.txt' }
& .\.venv\Scripts\python.exe -m pip install --no-compile -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -m pip freeze | Set-Content requirements.lock.txt
Write-Host 'Setup ready. Run .\start.ps1'
