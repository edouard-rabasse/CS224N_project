param(
    # Default location for virtual environments
    [string]$VenvPath = "C:\Users\rabas\python_envs\CS224N_project",

    # Persist environment variable for future sessions
    [switch]$Persist
)

Write-Host "Setting UV project environment to: $VenvPath"

# Create directory if it doesn't exist
if (!(Test-Path $VenvPath)) {
    New-Item -ItemType Directory -Path $VenvPath -Force | Out-Null
    Write-Host "Created directory: $VenvPath"
}

# Set environment variable for current session
$env:UV_PROJECT_ENVIRONMENT = $VenvPath

Write-Host "Running uv sync..."
uv sync

if ($LASTEXITCODE -ne 0) {
    Write-Error "uv sync failed."
    exit 1
}

Write-Host "uv sync completed successfully."

# Optionally persist the variable
if ($Persist) {
    Write-Host "Persisting UV_PROJECT_ENVIRONMENT for future sessions..."
    setx UV_PROJECT_ENVIRONMENT $VenvPath | Out-Null
    Write-Host "Environment variable saved."
}

Write-Host "Done."