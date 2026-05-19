param(
    [switch]$CreateStreamPETR,
    [switch]$CreateFlashOCC
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

if ($CreateFlashOCC) {
    conda env create -f (Join-Path $Root "envs\flashocc_env.yml")
    $MMDet3D = Join-Path $Root "third_party\FlashOCC-master\mmdetection3d"
    if (-not (Test-Path $MMDet3D)) {
        git clone https://github.com/open-mmlab/mmdetection3d.git $MMDet3D
        git -C $MMDet3D checkout v1.0.0rc4
    }
    conda run -n flashocc_env pip install -e $MMDet3D
    conda run -n flashocc_env pip install -e (Join-Path $Root "third_party\FlashOCC-master\projects")
}

if (-not $CreateStreamPETR -and -not $CreateFlashOCC) {
    Write-Host "Usage:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\setup_expert_envs.ps1 -CreateStreamPETR"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\setup_expert_envs.ps1 -CreateFlashOCC"
}
