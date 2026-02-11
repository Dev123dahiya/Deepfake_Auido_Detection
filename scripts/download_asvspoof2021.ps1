param(
  [ValidateSet('LA','PA','DF')] [string]$Track = 'LA',
  [string]$OutDir = (Resolve-Path (Join-Path $PSScriptRoot '..\data\asvspoof2021')).Path,
  [switch]$Extract
)

$ErrorActionPreference = 'Stop'

$trackDir = Join-Path $OutDir $Track
New-Item -ItemType Directory -Force -Path $trackDir | Out-Null

$keysUrlMap = @{
  LA = 'https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz'
  PA = 'https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz'
  DF = 'https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz'
}

$dataUrlMap = @{
  LA = @(
    'https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1'
  )
  PA = @(
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part00.tar.gz?download=1',
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part01.tar.gz?download=1',
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part02.tar.gz?download=1',
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part03.tar.gz?download=1',
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part04.tar.gz?download=1',
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part05.tar.gz?download=1',
    'https://zenodo.org/records/4834716/files/ASVspoof2021_PA_eval_part06.tar.gz?download=1'
  )
  DF = @(
    'https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1',
    'https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz?download=1',
    'https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz?download=1',
    'https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz?download=1'
  )
}

function Download-File([string]$Url, [string]$Dest) {
  Write-Host "Downloading: $Url"
  Write-Host " -> $Dest"
  if (Test-Path $Dest) { Write-Host "File exists, resuming if needed..." }
  & curl.exe -L -C - -o $Dest $Url
}

# Download keys/metadata
$keysUrl = $keysUrlMap[$Track]
$keysName = Split-Path $keysUrl -Leaf
$keysDest = Join-Path $trackDir $keysName
Download-File $keysUrl $keysDest

# Download data archives
foreach ($url in $dataUrlMap[$Track]) {
  $name = ($url -split '\?')[0] | Split-Path -Leaf
  $dest = Join-Path $trackDir $name
  Download-File $url $dest
}

if ($Extract) {
  Write-Host "Extracting archives..."
  Get-ChildItem -Path $trackDir -Filter '*.tar.gz' | ForEach-Object {
    Write-Host "Extracting $($_.Name)"
    & tar -xzf $_.FullName -C $trackDir
  }
}

Write-Host "Done. Files are in $trackDir"
