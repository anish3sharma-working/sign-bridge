@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -Command "$desktop = '%~1'; if ([string]::IsNullOrWhiteSpace($desktop)) { $desktop = [Environment]::GetFolderPath('Desktop') }; $desktop = [IO.Path]::GetFullPath($desktop); if (-not (Test-Path -LiteralPath $desktop -PathType Container)) { throw 'The requested Desktop folder does not exist.' }; $launcher = Join-Path '%~dp0' 'Launch SignBridge.bat'; $shortcutPath = Join-Path $desktop 'SignBridge.lnk'; $wsh = New-Object -ComObject WScript.Shell; if (Test-Path -LiteralPath $shortcutPath) { $old = $wsh.CreateShortcut($shortcutPath); if ($old.TargetPath -ne $launcher) { throw 'A different SignBridge.lnk already exists. Rename it before creating this shortcut.' } }; $shortcut = $wsh.CreateShortcut($shortcutPath); $shortcut.TargetPath = $launcher; $shortcut.WorkingDirectory = '%~dp0'; $shortcut.Description = 'Set up dependencies if needed, then launch SignBridge'; $shortcut.WindowStyle = 1; $shortcut.Save(); Write-Host ('Created: ' + $shortcutPath)"
if errorlevel 1 goto failed

echo.
echo The SignBridge desktop shortcut is ready.
if /I "%~2"=="/quiet" exit /b 0
pause
exit /b 0

:failed
echo.
echo Shortcut creation failed. No administrator access or execution-policy changes are needed.
if /I "%~2"=="/quiet" exit /b 1
pause
exit /b 1
