$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$cootPrefix = $env:MDNA_WINCOOT_ROOT
if (-not $cootPrefix) {
    throw 'Set MDNA_WINCOOT_ROOT to the WinCoot installation directory.'
}
$cootHome = Join-Path $projectRoot 'artifacts\rna_pilot\.coot-home'
$env:COOT_PREFIX = $cootPrefix
$env:COOT_GUILE_PREFIX = $cootPrefix.Replace('\', '/')
$env:COOT_SHARE = Join-Path $cootPrefix 'share'
$env:COOT_DATA_DIR = Join-Path $env:COOT_SHARE 'coot'
$env:COOT_STANDARD_RESIDUES = Join-Path $env:COOT_DATA_DIR 'standard-residues.pdb'
$env:COOT_REF_STRUCTS = Join-Path $env:COOT_DATA_DIR 'reference-structures'
$env:COOT_SCHEME_DIR = Join-Path $env:COOT_DATA_DIR 'scheme'
$env:COOT_REFMAC_LIB_DIR = Join-Path $env:COOT_DATA_DIR 'lib'
$env:COOT_PYTHON_DIR = Join-Path $cootPrefix 'lib\python3.12\site-packages\coot'
$env:COOT_HOME = $cootHome
$env:COOT_BACKUP_DIR = Join-Path $cootHome 'coot-backup'
$env:SYMINFO = Join-Path $env:COOT_SHARE 'syminfo.lib'
$env:PYTHONHOME = $cootPrefix
$env:PATH = "$cootPrefix\bin;$cootPrefix\lib;$env:PATH;$cootPrefix\bin\extras"

$arguments = @((Join-Path $PSScriptRoot 'generate_coot_rna_pilot.py')) + $args
& (Join-Path $cootPrefix 'bin\python.exe') @arguments
if ($LASTEXITCODE -ne 0) {
    throw "WinCoot Python exited with code $LASTEXITCODE"
}
