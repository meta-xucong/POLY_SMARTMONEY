$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
Set-Location $root

function Get-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @("py", "-3")
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }
    throw "Python launcher not found. Please install Python or add it to PATH."
}

$pycmd = Get-PythonCommand
$args = @(
    "$root\persistent_copytrade_runner.py",
    "launch",
    "--workdir", $root,
    "--config", "$root\copytrade_config.json",
    "--mode", "live",
    "--poll", "20",
    "--prefix", "persistent_live",
    "--session-name", "persistent_live_session.json"
)
$command = @($pycmd + $args)
$output = & $command[0] $command[1..($command.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "Launch failed with exit code $LASTEXITCODE"
}

try {
    $json = $output | ConvertFrom-Json
    if ($json.already_running) {
        Write-Host "Persistent live supervisor is already running." -ForegroundColor Yellow
        Write-Host "Session: $($json.session)"
        Write-Host "Supervisor PID: $($json.supervisor_pid)"
        Write-Host "Child PID: $($json.child_pid)"
    } else {
        Write-Host "Persistent live supervisor started." -ForegroundColor Green
        Write-Host "Session: $($json.session)"
        Write-Host "Supervisor PID: $($json.supervisor_pid)"
        Write-Host "Stdout log: $($json.stdout)"
        Write-Host "Stderr log: $($json.stderr)"
        Write-Host "Supervisor log: $($json.supervisor_log)"
    }
} catch {
    Write-Host $output
}
