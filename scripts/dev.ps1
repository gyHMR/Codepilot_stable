<#
.SYNOPSIS
Codepilot Docker development launcher.

.DESCRIPTION
Examples:
  .\scripts\dev.ps1                         # Build and run IM webhook service
  .\scripts\dev.ps1 -Mode cli               # Build and start interactive CLI
  .\scripts\dev.ps1 -Mode im -Transport longconn
#>
param(
    [ValidateSet("im", "cli")]
    [string]$Mode       = "im",
    [ValidateSet("webhook", "longconn")]
    [string]$Transport  = "webhook",
    [string]$ListenHost = "0.0.0.0",
    [int]   $Port       = 8787,
    [string]$Workspace  = "/workspace",
    [string]$LogLevel   = "debug"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:CODEPILOT_TRANSPORT = $Transport
$env:CODEPILOT_HOST = $ListenHost
$env:CODEPILOT_PORT = [string]$Port
$env:CODEPILOT_WORKSPACE = $Workspace
$env:CODEPILOT_LOG_LEVEL = $LogLevel

if ($Mode -eq "im") {
    Write-Host "[dev] Building and starting Codepilot IM service with Docker Compose ..." -ForegroundColor Green
    docker compose up --build codepilot-im
} else {
    Write-Host "[dev] Building and starting Codepilot CLI with Docker Compose ..." -ForegroundColor Green
    docker compose build codepilot-cli
    docker compose run --rm codepilot-cli
}
