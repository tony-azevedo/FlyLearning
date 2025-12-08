param(
    [string]$BasePath = "D:\Data\251107\251107_F1_C1",
    [switch]$Delete,          # actually delete files (otherwise dry-run)
    [switch]$Recycle          # send to Recycle Bin (requires -Delete)
)

Write-Host "Scanning for AcquireWithEpiFeedback_ContRaw_*.bin under:" -ForegroundColor Cyan
Write-Host "  $BasePath`n"

if (-not (Test-Path -LiteralPath $BasePath)) {
    Write-Error "Base path not found: $BasePath"
    exit 1
}

# Find only the target .bin files recursively
$files = Get-ChildItem -LiteralPath $BasePath -Recurse -File -Filter "AcquireWithEpiFeedback_ContRaw_*.bin" -ErrorAction SilentlyContinue

if (-not $files -or $files.Count -eq 0) {
    Write-Host "No matching .bin files found. Nothing to do." -ForegroundColor Yellow
    exit 0
}

# Calculate total size
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
$totalMB = [Math]::Round($totalBytes / 1MB, 2)
$totalGB = [Math]::Round($totalBytes / 1GB, 3)
Write-Host ("Found {0} file(s), total size ~{1} MB (~{2} GB)." -f $files.Count, $totalMB, $totalGB) -ForegroundColor Green

# Dry-run preview
if (-not $Delete) {
    Write-Host "`nDry run (no files deleted). These would be removed:" -ForegroundColor Yellow
    $files | Sort-Object Length -Descending | ForEach-Object {
        $sizeMB = [Math]::Round($_.Length / 1MB, 2)
        Write-Host ("  {0}  ({1} MB)" -f $_.FullName, $sizeMB)
    }

    Write-Host "`nTotal size that would be deleted: $totalMB MB (~$totalGB GB)"
    Write-Host "`nTo delete, re-run with:  .\Delete-ContRawBins.ps1 -Delete"
    Write-Host "To send to Recycle Bin instead:  .\Delete-ContRawBins.ps1 -Delete -Recycle"
    exit 0
}

# If deleting, optionally use Recycle Bin
if ($Recycle) {
    Add-Type -AssemblyName Microsoft.VisualBasic
    $UI = [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs
    $REC = [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin

    $errors = @()
    foreach ($f in $files) {
        try {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($f.FullName, $UI, $REC)
            Write-Host "Recycled: $($f.FullName)"
        } catch {
            $errors += $_
            Write-Warning "Failed to recycle: $($f.FullName) -> $($_.Exception.Message)"
        }
    }

    Write-Host "`nDone. Recycled $($files.Count - $errors.Count) file(s)."
    if ($errors.Count -gt 0) { Write-Warning ("Errors: {0}" -f $errors.Count) }
    exit 0
}
else {
    # Permanent delete
    $errors = @()
    foreach ($f in $files) {
        try {
            Remove-Item -LiteralPath $f.FullName -Force
            Write-Host "Deleted: $($f.FullName)"
        } catch {
            $errors += $_
            Write-Warning "Failed to delete: $($f.FullName) -> $($_.Exception.Message)"
        }
    }

    Write-Host "`nDone. Deleted $($files.Count - $errors.Count) file(s)."
    if ($errors.Count -gt 0) { Write-Warning ("Errors: {0}" -f $errors.Count) }
}
