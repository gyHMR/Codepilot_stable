$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$web = Join-Path $repo "web"
$target = Join-Path $repo "src\codepilot\interfaces\web\static"
$expected = [System.IO.Path]::GetFullPath($target)

Push-Location $web
try {
    npm ci
    npm run build
} finally {
    Pop-Location
}

$resolved = [System.IO.Path]::GetFullPath($target)
if ($resolved -ne $expected -or -not $resolved.EndsWith("src\codepilot\interfaces\web\static")) {
    throw "Refusing to replace unexpected Web static directory: $resolved"
}
if (Test-Path -LiteralPath $resolved) {
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
New-Item -ItemType Directory -Path $resolved | Out-Null
Copy-Item -Path (Join-Path $web "dist\*") -Destination $resolved -Recurse -Force
