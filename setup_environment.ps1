param(
    [ValidateSet('Control', 'VLM', 'OpenVLA', 'All')]
    [string]$Mode = 'Control'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw '未找到 uv，请先安装 uv 后重新运行此脚本。'
}

function Invoke-Uv {
    param([string[]]$Arguments)
    & $uvCommand.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "环境配置失败，uv 退出码为 $LASTEXITCODE"
    }
}

if ($Mode -in @('Control', 'VLM', 'All')) {
    $controlPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $controlPython)) {
        Invoke-Uv -Arguments @('venv', '--python', '3.11', '.venv')
    }
    Invoke-Uv -Arguments @('pip', 'install', '--python', $controlPython, '--link-mode', 'copy', '-r', 'requirements.txt')
    if ($Mode -in @('VLM', 'All')) {
        Invoke-Uv -Arguments @('pip', 'install', '--python', $controlPython, '--link-mode', 'copy', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124')
        Invoke-Uv -Arguments @('pip', 'install', '--python', $controlPython, '--link-mode', 'copy', '-r', 'requirements-vlm.txt')
    }
    Invoke-Uv -Arguments @('pip', 'check', '--python', $controlPython)
}

if ($Mode -in @('OpenVLA', 'All')) {
    $openvlaPython = Join-Path $PSScriptRoot '.venv-openvla\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $openvlaPython)) {
        Invoke-Uv -Arguments @('venv', '--python', '3.10', '.venv-openvla')
    }
    Invoke-Uv -Arguments @('pip', 'install', '--python', $openvlaPython, '--link-mode', 'copy', 'torch==2.2.0', 'torchvision==0.17.0', '--index-url', 'https://download.pytorch.org/whl/cu121')
    Invoke-Uv -Arguments @('pip', 'install', '--python', $openvlaPython, '--link-mode', 'copy', '-r', 'requirements-openvla.txt')
    Invoke-Uv -Arguments @('pip', 'check', '--python', $openvlaPython)
}

Write-Output "环境配置完成，模式为 $Mode"
