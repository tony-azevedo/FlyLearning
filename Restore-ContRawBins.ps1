<#
.SYNOPSIS
Restore continuous .bin files for a given FlyLearning data folder using rclone.

.DESCRIPTION
- Copies only files matching: AcquireWithEpiFeedback_ContRaw_*.bin
- Sources from a given rclone remote that mirrors D:\Data structure
- Works for current directory, a full path, or a leaf folder name like "251107_F1_C1"
- Optional dry-run and list-only modes

.EXAMPLES
  # Restore .bin files for the *current* directory:
  .\restore_bin_files.ps1

  # Restore for an explicit path:
  .\restore_bin_files.ps1 -Path "D:\Data\251107\251107_F1_C1"

  # Restore by leaf name (find under D:\Data):
  .\restore_bin_files.ps1 -Path "251107_F1_C1"

  # Preview remote contents only:
  .\restore_bin_files.ps1 -ListOnly

  # Dry run (no writes):
  .\restore_bin_files.ps1 -DryRun
#>

param(
    # A path under D:\Data (or just a leaf like 251107_F1_C1). If omitted, uses current directory.
    [string]$Path,

    # rclone remote that mirrors D:\Data (adjust to your remote/path)
    [string]$Remote = "gdrive:FlyLearningBackups",

    # Root of the local dataset
    [string]$LocalRoot = "D:\Data",

    # Only show what is available remotely (no copying)
    [switch]$ListOnly,

    # Do not write; pass --dry-run through to rclone
    [switch]$DryRun
)

# --- helpers ---
function Assert-Rclone {
    $r = Get-Command rclone -ErrorAction SilentlyContinue
    if (-not $r) { throw "rclone not found in PATH. Please install or add to PATH." }
}

function Get-TargetDirs {
    param(
        [string]$PathOrLeaf,
        [string]$LocalRoot
    )

    # If no path specified, use current directory
    if ([string]::IsNullOrWhiteSpace($PathOrLeaf)) {
        $here = (Get-Location).Path
        return ,(Get-Item -LiteralPath $here)
    }

    # If it's a full/relative path that exists, use it directly
    if (Test-Path -LiteralPath $PathOrLeaf) {
        $item = Get-Item -LiteralPath $PathOrLeaf
        if (-not $item.PSIsContainer) { throw "Path is not a directory: $PathOrLeaf" }
        return ,$item
    }

    # Otherwise treat as a leaf name and search under LocalRoot
    $leaf = $PathOrLeaf
    if (-not (Test-Path -LiteralPath $LocalRoot)) {
        throw "Local root not found: $LocalRoot"
    }
    $matches = Get-ChildItem -LiteralPath $LocalRoot -Recurse -Directory -Filter $leaf -ErrorAction SilentlyContinue
    if (-not $matches -or $matches.Count -eq 0) {
        throw "No directories named '$leaf' found under $LocalRoot"
    }
    return $matches
}

function Get-RemoteDirForLocal {
    param(
        [string]$LocalDir,
        [string]$LocalRoot,
        [string]$Remote
    )
    $full = (Get-Item -LiteralPath $LocalDir).FullName
    $root = (Get-Item -LiteralPath $LocalRoot).FullName

    if ($full.ToLower().StartsWith($root.ToLower())) {
        $rel = $full.Substring($root.Length).TrimStart('\','/')
    } else {
        # If LocalDir is outside LocalRoot, just map to the same name under the remote root
        $rel = Split-Path -Leaf $full
    }

    # rclone likes forward slashes
    $rel = $rel -replace '\\','/'
    $remoteDir = ($Remote.TrimEnd('/')) + "/" + ($rel.TrimStart('/'))
    return $remoteDir
}

# --- main ---
try {
    Assert-Rclone

    $targets = Get-TargetDirs -PathOrLeaf $Path -LocalRoot $LocalRoot
    $pattern = "AcquireWithEpiFeedback_ContRaw_*.bin"

    foreach ($t in $targets) {
        $localDir  = $t.FullName
        $remoteDir = Get-RemoteDirForLocal -LocalDir $localDir -LocalRoot $LocalRoot -Remote $Remote

        Write-Host ""
        Write-Host "Local : $localDir" -ForegroundColor Cyan
        Write-Host "Remote: $remoteDir" -ForegroundColor Cyan
        Write-Host "Filter: $pattern"   -ForegroundColor Cyan

        $baseArgs = @()
        if ($DryRun)   { $baseArgs += "--dry-run" }

        if ($ListOnly) {
            # Show what’s available remotely that matches the pattern
            Write-Host "Listing remote files..." -ForegroundColor Yellow
            & rclone ls "$remoteDir" --include "$pattern" @baseArgs
            continue
        }

        # Copy only the matching .bin files from remote → local
        $args = @(
            "copy", "$remoteDir", "$localDir",
            "--include", $pattern,
            "--create-empty-src-dirs",
            "--progress",
            "--transfers=8"
        ) + $baseArgs

        Write-Host "Running: rclone $($args -join ' ')" -ForegroundColor Gray
        & rclone @args
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "rclone returned non-zero exit code ($LASTEXITCODE) for: $localDir"
        } else {
            Write-Host "Done for: $localDir" -ForegroundColor Green
        }
    }

} catch {
    Write-Error $_.Exception.Message
    exit 1
}
