param([switch]$SkipDownload)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $taskPython)) { throw 'Set up .venv using README.md first.' }
function Invoke-TaskPython {
    & $taskPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
if (-not $SkipDownload) {
    if (-not (Test-Path 'data/raw/cwfid/.git')) {
        git clone https://github.com/cwfid/dataset.git data/raw/cwfid
        if ($LASTEXITCODE -ne 0) { throw 'CWFID clone failed' }
        git -C data/raw/cwfid checkout 36290d0912a032f0bd1a5678ed55457ddafde217
        if ($LASTEXITCODE -ne 0) { throw 'CWFID revision checkout failed' }
    }
    if (-not (Test-Path 'sources/clinicdb_drive_index.json')) {
        Invoke-TaskPython scripts/download_clinicdb.py
    } else {
        Invoke-TaskPython scripts/restore_clinicdb_mirror.py
    }
}
if (-not (Test-Path 'data/manifests/clinicdb.json')) { Invoke-TaskPython scripts/prepare_data.py clinicdb }
if (-not (Test-Path 'data/manifests/cwfid.json')) { Invoke-TaskPython scripts/prepare_data.py cwfid }
Invoke-TaskPython -m unittest discover -s tests -v
foreach ($taskConfig in @('configs/clinicdb_reference.json', 'configs/cwfid_domain.json')) {
    $taskRun = (Get-Content $taskConfig | ConvertFrom-Json).output
    if (-not (Test-Path "$taskRun/metrics.json")) {
        if (Test-Path "$taskRun/last.pt") {
            Invoke-TaskPython -m screening.run train --config $taskConfig --resume
        } else {
            Invoke-TaskPython -m screening.run train --config $taskConfig
        }
    }
}
Invoke-TaskPython scripts/build_deliverables.py
Invoke-TaskPython scripts/verify_deliverables.py
