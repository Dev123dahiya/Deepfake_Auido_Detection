param(
  [string]$LaRoot = "C:\Users\KHUSHI\Downloads\deepfake_code\other_files\data\asvspoof2021\LA",
  [string]$OutRoot = "C:\Users\KHUSHI\Downloads\deepfake_code\important_files\dataset\asvspoof2021_la_eval"
)

$ErrorActionPreference = "Stop"

$keysArchive = Join-Path $LaRoot "LA-keys-full.tar.gz"
$evalArchive = Join-Path $LaRoot "ASVspoof2021_LA_eval.tar.gz"
$metaPath = Join-Path $LaRoot "keys\LA\CM\trial_metadata.txt"
$flacDir = Join-Path $LaRoot "ASVspoof2021_LA_eval\flac"
$realDir = Join-Path $OutRoot "real"
$fakeDir = Join-Path $OutRoot "fake"

if (-not (Test-Path $metaPath)) {
  if (-not (Test-Path $keysArchive)) { throw "Missing keys archive: $keysArchive" }
  tar -xzf $keysArchive -C $LaRoot
}

if (-not (Test-Path $flacDir)) {
  if (-not (Test-Path $evalArchive)) { throw "Missing eval archive: $evalArchive" }
  tar -xzf $evalArchive -C $LaRoot
}

New-Item -ItemType Directory -Force -Path $realDir, $fakeDir | Out-Null

$labelMap = @{}
Get-Content $metaPath | ForEach-Object {
  if ([string]::IsNullOrWhiteSpace($_)) { return }
  $parts = $_ -split "\s+"
  if ($parts.Count -ge 8) {
    $utt = $parts[1]
    $label = $parts[5]
    $split = $parts[7]
    if ($split -eq "eval") {
      $labelMap[$utt] = $label
    }
  }
}

$linked = 0
$copied = 0
$missingLabel = 0

Get-ChildItem -Path $flacDir -Filter "*.flac" | ForEach-Object {
  $id = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
  if (-not $labelMap.ContainsKey($id)) {
    $missingLabel++
    return
  }

  $destDir = if ($labelMap[$id] -eq "bonafide") { $realDir } else { $fakeDir }
  $dest = Join-Path $destDir $_.Name
  if (Test-Path $dest) { return }

  try {
    New-Item -ItemType HardLink -Path $dest -Target $_.FullName -ErrorAction Stop | Out-Null
    $linked++
  } catch {
    Copy-Item -Path $_.FullName -Destination $dest -Force
    $copied++
  }
}

$expectedEval = (Get-Content $metaPath | Where-Object { ($_ -split "\s+")[7] -eq "eval" }).Count
$realCount = (Get-ChildItem $realDir -Filter "*.flac").Count
$fakeCount = (Get-ChildItem $fakeDir -Filter "*.flac").Count
$total = $realCount + $fakeCount

Write-Host "Output dataset: $OutRoot"
Write-Host "Expected eval entries in metadata: $expectedEval"
Write-Host "Prepared real/fake files: $realCount / $fakeCount (total: $total)"
Write-Host "Hardlinked: $linked | Copied: $copied | MissingLabel: $missingLabel"

if ($total -lt $expectedEval) {
  Write-Warning "Prepared total is smaller than expected metadata count. Archive may be incomplete/corrupted."
}

