# Installs bitty on Windows from the prebuilt binaries and keeps it up to date.
#
#   irm https://raw.githubusercontent.com/askscience/bitty/qwen-llama-cpp/scripts/install.ps1 | iex
#
# What it does:
#   - downloads bitty.exe and the stage engine bitty-stage.exe into %LOCALAPPDATA%\bitty\bin
#   - adds that folder to your PATH
#   - registers a scheduled task that looks for a new build every minute, installs it,
#     restarts the node if one was running, and starts the node again after a reboot
#
# Nothing here needs administrator rights. The same file is installed as bitty-update.ps1,
# which is what the scheduled task runs.
#
#   install.ps1                 install or reinstall
#   install.ps1 -NoAutoUpdate   install without the scheduled task
#   install.ps1 -Update         check once and update if there is a new build (the task's job)
#   install.ps1 -Uninstall      remove the binaries and the task; models and settings stay
#
# Environment (the only way to pass options to the one-line form above):
#   BITTY_HOME            where bitty lives         (default %LOCALAPPDATA%\bitty)
#   BITTY_CHANNEL         release tag to follow     (default nightly)
#   BITTY_NO_AUTO_UPDATE  set to 1 for -NoAutoUpdate
#   BITTY_ARCHIVE         install from this local zip, no download and no checksum (for tests)
param(
    [switch]$Update,
    [switch]$Uninstall,
    [switch]$NoAutoUpdate
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Repo = 'askscience/bitty'
$Channel = if ($env:BITTY_CHANNEL) { $env:BITTY_CHANNEL } else { 'nightly' }
$BaseUrl = if ($env:BITTY_BASE_URL) { $env:BITTY_BASE_URL } else { "https://github.com/$Repo/releases/download/$Channel" }
$Prefix = if ($env:BITTY_HOME) { $env:BITTY_HOME } else { Join-Path $env:LOCALAPPDATA 'bitty' }
$BinDir = Join-Path $Prefix 'bin'
$Bitty = Join-Path $BinDir 'bitty.exe'
$Updater = Join-Path $BinDir 'bitty-update.ps1'
$VersionFile = Join-Path $Prefix 'VERSION'
$LogFile = Join-Path $Prefix 'update.log'
$TaskName = 'bitty-update'
$IntervalMinutes = 1
$Name = 'bitty-windows-x86_64'

function Write-Log([string]$Message) {
    Add-Content -Path $LogFile -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
}

function Get-InstalledVersion {
    if (Test-Path $VersionFile) { return (Get-Content $VersionFile -Raw).Trim() }
    return ''
}

function Assert-Platform {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'there is no prebuilt bitty for this processor yet; build it from source (see engine/README.md)'
    }
}

# Runs bitty and returns its exit code. The output is deliberately not captured: a node that
# `bitty restart` starts inherits the capture pipe and keeps it open for as long as it lives,
# and PowerShell would wait on that pipe forever, with the update lock held.
function Invoke-Bitty([string[]]$Arguments) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Bitty
    $info.Arguments = ($Arguments -join ' ')
    $info.UseShellExecute = $false
    $process = [Diagnostics.Process]::Start($info)
    # two minutes are far more than any of these commands needs
    if (-not $process.WaitForExit(120000)) {
        Write-Log ("bitty {0} did not finish in two minutes" -f ($Arguments -join ' '))
        return 1
    }
    $code = $process.ExitCode
    if ($code -ne 0) { Write-Log ("bitty {0} failed with exit code {1}" -f ($Arguments -join ' '), $code) }
    return $code
}

# A running .exe cannot be overwritten on Windows, but it can be renamed: move the old file
# aside, put the new one in its place, and delete the old one once the node has restarted.
function Install-File([string]$Source, [string]$Target) {
    if (Test-Path $Target) {
        $old = "$Target.old-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
        Move-Item -Path $Target -Destination $old -Force
    }
    Move-Item -Path $Source -Destination $Target -Force
}

function Remove-OldFiles {
    Get-ChildItem -Path $BinDir -Filter '*.old-*' -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
}

# Downloads the current build when it differs from the installed one and swaps the files in.
# Returns $true when something was installed.
function Update-Binaries {
    $work = Join-Path ([IO.Path]::GetTempPath()) ("bitty-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    try {
        if ($env:BITTY_ARCHIVE) {
            $archive = $env:BITTY_ARCHIVE
        } else {
            $latest = (Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/VERSION").Content
            if ($latest -is [byte[]]) { $latest = [Text.Encoding]::UTF8.GetString($latest) }
            $latest = "$latest".Trim()
            if (-not $latest) { throw 'the release has an empty VERSION file' }
            $complete = (Test-Path $Bitty) -and (Test-Path (Join-Path $BinDir 'bitty-stage.exe'))
            if ($latest -eq (Get-InstalledVersion) -and $complete) { return $false }

            $archive = Join-Path $work "$Name.zip"
            Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$Name.zip" -OutFile $archive
            $sumsFile = Join-Path $work 'SHA256SUMS'
            Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/SHA256SUMS" -OutFile $sumsFile
            $line = Get-Content $sumsFile | Where-Object { $_ -match "\s\*?$([regex]::Escape("$Name.zip"))$" } | Select-Object -First 1
            if (-not $line) { throw "SHA256SUMS has no line for $Name.zip" }
            $expected = ($line -split '\s+')[0].ToLower()
            $actual = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()
            # a build that is being published can have a new archive next to old checksums:
            # refuse it now, the next run sees the finished release
            if ($expected -ne $actual) { throw "the checksum of $Name.zip does not match; nothing was changed" }
        }

        Expand-Archive -Path $archive -DestinationPath (Join-Path $work 'new') -Force
        $new = Join-Path (Join-Path $work 'new') $Name
        foreach ($file in 'bitty.exe', 'bitty-stage.exe', 'install.ps1', 'VERSION') {
            if (-not (Test-Path (Join-Path $new $file))) { throw "the archive has no $file" }
        }
        & (Join-Path $new 'bitty.exe') version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'the new bitty does not run on this machine; nothing was changed' }

        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
        Install-File (Join-Path $new 'bitty.exe') $Bitty
        Install-File (Join-Path $new 'bitty-stage.exe') (Join-Path $BinDir 'bitty-stage.exe')
        Copy-Item -Path (Join-Path $new 'install.ps1') -Destination $Updater -Force
        Copy-Item -Path (Join-Path $new 'VERSION') -Destination $VersionFile -Force
        return $true
    } finally {
        Remove-Item -Path $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Add-ToUserPath {
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($current) { $parts = $current -split ';' | Where-Object { $_ } }
    if ($parts -notcontains $BinDir) {
        [Environment]::SetEnvironmentVariable('Path', (($parts + $BinDir) -join ';'), 'User')
    }
    if (($env:Path -split ';') -notcontains $BinDir) { $env:Path = "$env:Path;$BinDir" }
}

function Remove-FromUserPath {
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $current) { return }
    $parts = $current -split ';' | Where-Object { $_ -and $_ -ne $BinDir }
    [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
}

function Register-UpdateTask {
    $powershell = Join-Path $PSHOME 'powershell.exe'
    $command = "`"$powershell`" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Updater`" -Update"
    $conhost = Join-Path $env:SystemRoot 'System32\conhost.exe'
    if (Test-Path $conhost) {
        # a console program started by the task scheduler flashes a window every time;
        # a headless console host does not
        $action = New-ScheduledTaskAction -Execute $conhost -Argument "--headless $command"
    } else {
        $action = New-ScheduledTaskAction -Execute $powershell -Argument "-WindowStyle Hidden -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Updater`" -Update"
    }
    $user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $atLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
    $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $atLogon, $repeat -Settings $settings `
        -Principal $principal -Description 'Updates bitty and keeps its node running' -Force | Out-Null
}

# An installation made by an older build keeps that build's task. The updater brings it to the
# current interval; an installation without auto update has no task and is left alone.
function Update-TaskInterval {
    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if (-not $task) { return }
        $wanted = [Xml.XmlConvert]::ToString((New-TimeSpan -Minutes $IntervalMinutes))
        $current = $task.Triggers | ForEach-Object { $_.Repetition.Interval } | Where-Object { $_ }
        if ($current -contains $wanted) { return }
        Write-Log "setting the update check to every $IntervalMinutes minute(s)"
        Register-UpdateTask
    } catch {
        Write-Log "the scheduled task could not be changed: $($_.Exception.Message)"
    }
}

function Unregister-UpdateTask {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

function Invoke-Install {
    Assert-Platform
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    if (Update-Binaries) {
        Write-Host "installed bitty $(Get-InstalledVersion) in $BinDir"
        # a node that was running keeps the old binary until it is restarted
        Invoke-Bitty @('restart', '--if-started') | Out-Null
        Remove-OldFiles
    } else {
        Write-Host "bitty $(Get-InstalledVersion) is already the current build"
    }
    Add-ToUserPath
    if ($NoAutoUpdate -or $env:BITTY_NO_AUTO_UPDATE -eq '1') {
        Unregister-UpdateTask
        Write-Host "auto update: off; run '$Updater -Update' to update by hand"
    } else {
        try {
            Register-UpdateTask
            Write-Host "auto update: scheduled task '$TaskName' every $IntervalMinutes min while you are logged in"
        } catch {
            Write-Host "auto update: could not register the scheduled task ($($_.Exception.Message))"
            Write-Host "  run '$Updater -Update' whenever you want the current build"
        }
    }
    Write-Host ''
    Write-Host 'Open a new terminal so that PATH is picked up, then:'
    Write-Host '  first machine:   bitty share home        (prints the invite)'
    Write-Host "  other machines:  bitty connect 'INVITE' --name home"
    Write-Host '  then, anywhere:  bitty run qwen3.5-35b-a3b "hello"'
}

# What the scheduled task runs. Silent: what happens goes to update.log.
function Invoke-Update {
    New-Item -ItemType Directory -Path $Prefix -Force | Out-Null
    # one run at a time: a slow download must not meet the next tick
    $mutex = New-Object Threading.Mutex($false, 'Local\bitty-update')
    if (-not $mutex.WaitOne(0)) { return }
    try {
        if ((Test-Path $LogFile) -and (Get-Item $LogFile).Length -gt 1MB) {
            $tail = Get-Content $LogFile -Tail 200
            Set-Content -Path $LogFile -Value $tail
        }
        $before = Get-InstalledVersion
        try {
            if (Update-Binaries) {
                Write-Log "updated $before -> $(Get-InstalledVersion)"
                Invoke-Bitty @('restart', '--if-started') | Out-Null
            }
        } catch {
            # a failed download must not stop the node from being looked after below
            Write-Log "update failed: $($_.Exception.Message)"
        }
        Remove-OldFiles
        # after a reboot or a crash: bring back the node that was running. Does nothing when
        # bitty was stopped with `bitty stop` or never started.
        if (Test-Path $Bitty) { Invoke-Bitty @('start', '--if-started') | Out-Null }
        Update-TaskInterval
    } finally {
        $mutex.ReleaseMutex()
    }
}

function Invoke-Uninstall {
    if (Test-Path $Bitty) { Invoke-Bitty @('stop') | Out-Null }
    Unregister-UpdateTask
    Remove-FromUserPath
    Remove-Item -Path $Prefix -Recurse -Force -ErrorAction SilentlyContinue
    $data = if ($env:BITTY_DATA_DIR) { $env:BITTY_DATA_DIR } else { Join-Path $env:USERPROFILE '.bitty' }
    Write-Host "bitty was removed. Models and settings are still in $data."
}

if ($Update) { Invoke-Update }
elseif ($Uninstall) { Invoke-Uninstall }
else { Invoke-Install }
