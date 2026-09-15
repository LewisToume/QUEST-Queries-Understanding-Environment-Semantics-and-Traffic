param(
    [switch]$CreateStreamPETR
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found. Install Miniconda/Anaconda or run this inside a conda-enabled shell."
}

if ($CreateStreamPETR) {
    conda env create -f (Join-Path $Root "envs\streampetr_env.yml")
    $MMDet3D = Join-Path $Root "third_party\StreamPETR-main\mmdetection3d"
    if (-not (Test-Path $MMDet3D)) {
        git clone https://github.com/open-mmlab/mmdetection3d.git $MMDet3D
        git -C $MMDet3D checkout v1.0.0rc6
    }
    conda run -n streampetr_env pip install -e $MMDet3D
}

if (-not $CreateStreamPETR) {
    Write-Host "Usage:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\setup_expert_envs.ps1 -CreateStreamPETR"
}
