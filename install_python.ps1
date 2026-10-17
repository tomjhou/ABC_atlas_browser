# Downloads and installs the newest stable Python 3.13 for the current user (no administrator rights needed), then writes the
# path of its python.exe to the file given as -ResultFile. Called by ensure_python.bat when no Python is found.
param([Parameter(Mandatory = $true)][string]$ResultFile)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # (the progress bar makes downloads in Windows PowerShell very slow)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (Test-Path $ResultFile) { Remove-Item $ResultFile -Force }

# The Python minor version to install. Pinned to one the libraries these programs use are known to work with: a brand-new
# Python often has no prebuilt libraries (numpy, h5py, ...) for a while. Raise it when they have caught up.
$PythonMajor = 3
$PythonMinor = 13

$base = "https://www.python.org/ftp/python"
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "amd64" }

Write-Host "Looking up the newest version of Python $PythonMajor.$PythonMinor..."
$listing = (Invoke-WebRequest -Uri "$base/" -UseBasicParsing).Content
$versions = [regex]::Matches($listing, 'href="(\d+\.\d+\.\d+)/"') | ForEach-Object { [version]$_.Groups[1].Value } |
    Sort-Object -Descending -Unique

$installerUrl = $null
foreach ($v in $versions) {
    if ($v.Major -ne $PythonMajor -or $v.Minor -ne $PythonMinor) { continue }
    $url = "$base/$v/python-$v-$arch.exe"   # (only final releases have this name; the last patch releases of an old version are source-only)
    try {
        Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing | Out-Null
        $installerUrl = $url
        break
    } catch { }
}
if (-not $installerUrl) { throw "Couldn't find a Python installer at $base" }

$installer = Join-Path $env:TEMP (Split-Path $installerUrl -Leaf)
Write-Host "Downloading $installerUrl ..."
Invoke-WebRequest -Uri $installerUrl -OutFile $installer -UseBasicParsing

Write-Host "Installing Python (this takes a minute or two)..."
$p = Start-Process -FilePath $installer -Wait -PassThru -ArgumentList @(
    "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1", "Include_test=0", "Include_doc=0")
Remove-Item $installer -Force -ErrorAction SilentlyContinue
if ($p.ExitCode -ne 0) { throw "The Python installer failed (exit code $($p.ExitCode))" }

$found = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Programs\Python\Python3*\python.exe") -ErrorAction SilentlyContinue |
    Sort-Object { [int]($_.Directory.Name -replace '\D', '') } -Descending | Select-Object -First 1
if (-not $found) { throw "Python was installed, but its python.exe could not be found" }
Set-Content -Path $ResultFile -Value $found.FullName -Encoding ascii
