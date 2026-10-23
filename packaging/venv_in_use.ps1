# Used by build_atlas_browser.bat and build_assembler.bat before they install libraries into the .venv: exits with 1 (after saying which programs) if any
# program is running from that virtual environment. Windows keeps the libraries a running program has loaded locked, so pip cannot replace them and
# leaves the environment half changed.
param([string]$VenvDir)

$resolved = Resolve-Path -LiteralPath $VenvDir -ErrorAction SilentlyContinue
if (-not $resolved) { exit 0 }                      # (no environment yet: nothing can be running from it)
$venv = $resolved.Path.TrimEnd('\')
$prefix = $venv + '\'

$running = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and (
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and $_.CommandLine.IndexOf($prefix, [System.StringComparison]::OrdinalIgnoreCase) -ge 0))
}
if (-not $running) { exit 0 }

Write-Host ""
Write-Host "These programs are running from the virtual environment ($venv):"
foreach ($p in $running) {
    $cmd = if ($p.CommandLine) { $p.CommandLine } else { $p.Name }
    if ($cmd.Length -gt 100) { $cmd = $cmd.Substring(0, 100) + "..." }
    Write-Host ("  {0,7}  {1}" -f $p.ProcessId, $cmd)
}
Write-Host ""
Write-Host "Windows keeps the libraries they use locked, so the build could not replace them and would leave the environment half changed."
Write-Host "Close them (the atlas browser, the assembler, any window started by a RUN_*.bat file) and run this again."
exit 1
