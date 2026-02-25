# Activate the ipd-marl virtual environment and set UV_PROJECT_ENVIRONMENT.
# Usage (from PowerShell):  . .\activate_env.ps1   (note the leading dot)

$env:UV_PROJECT_ENVIRONMENT = "C:\Users\rabas\python_envs\CS224N_project"
& "C:\Users\rabas\python_envs\CS224N_project\Scripts\Activate.ps1"

Write-Host ""
Write-Host "[CS224N_project] Virtual environment activated." -ForegroundColor Green
Write-Host "  Python : C:\Users\rabas\python_envs\CS224N_project\Scripts\python.exe"
Write-Host "  UV env : $env:UV_PROJECT_ENVIRONMENT"
Write-Host ""
