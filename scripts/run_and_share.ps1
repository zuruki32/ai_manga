<#
Run the manga-ai pipeline (or one stage), snapshot a small result set into
results/, and push it so a reviewer can pull and inspect. Windows PowerShell.

Examples:
  .\scripts\run_and_share.ps1                               # full run, defaults below
  .\scripts\run_and_share.ps1 -Stage translate             # translation only (fast)
  .\scripts\run_and_share.ps1 -Stage clean -Force          # re-clean, ignore cache
  .\scripts\run_and_share.ps1 -Chapter "data/chapters/Ch.133 - ..." -Config hybrid_qwen_gemma_it
  .\scripts\run_and_share.ps1 -NoPush                       # run + collect, don't push
#>
param(
  [string]$Chapter = "data/chapters/Ch.133 - Suddenly Became A Princess One Day",
  [string]$Config  = "hybrid_qwen_gemma_it",
  [ValidateSet("full","detect","ocr","translate","mask","clean")]
  [string]$Stage   = "full",
  [int]$Samples    = 6,
  [switch]$Force,
  [switch]$NoPush,
  [string]$Python  = "python"
)
$ErrorActionPreference = "Continue"

# cd to repo root (parent of this script dir)
Set-Location (Join-Path $PSScriptRoot "..")

$cmdMap = @{ full="process"; detect="detect"; ocr="ocr"; translate="translate"; mask="mask"; clean="clean" }
$Cmd = $cmdMap[$Stage]
$Branch = (git rev-parse --abbrev-ref HEAD) 2>$null
if (-not $Branch) { $Branch = "main" }
$forceArg = @(); if ($Force) { $forceArg = @("--force") }

Write-Host "== repo:    $(Get-Location)"
Write-Host "== branch:  $Branch"
Write-Host "== stage:   $Stage  (manga-ai $Cmd)"
Write-Host "== config:  $Config"
Write-Host "== chapter: $Chapter"

# 1) pull latest code first
Write-Host "-- git pull --rebase --autostash origin $Branch"
git pull --rebase --autostash origin $Branch
if ($LASTEXITCODE -ne 0) { Write-Host "!! pull failed (offline or conflict) - continuing with local code" }

# 2) run the pipeline, capturing a log
$log = [System.IO.Path]::GetTempFileName()
Write-Host "-- running: $Python -m manga_ai $Cmd `"$Chapter`" -c $Config $($forceArg -join ' ')"
& $Python -m manga_ai $Cmd $Chapter -c $Config @forceArg 2>&1 | Tee-Object -FilePath $log
$runRc = $LASTEXITCODE
Write-Host "-- pipeline exit code: $runRc"

# 3) collect a small shareable snapshot
Write-Host "-- collecting results"
& $Python scripts/collect_results.py --chapter $Chapter --config $Config --out results/latest --samples $Samples
if (Test-Path results/latest) { Copy-Item $log results/latest/run.txt -Force -ErrorAction SilentlyContinue }
Remove-Item $log -ErrorAction SilentlyContinue

# 4) commit + push
if (-not $NoPush) {
  git add -A results/
  git diff --cached --quiet
  if ($LASTEXITCODE -eq 0) {
    Write-Host "-- nothing new in results/ to commit"
  } else {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    git commit -m "run results: $Stage / $Config @ $stamp" | Out-Null
    Write-Host "-- pushing results to origin/$Branch"
    for ($i=1; $i -le 4; $i++) {
      git push origin "HEAD:$Branch"
      if ($LASTEXITCODE -eq 0) {
        Write-Host "-- pushed. Reviewer: git pull; open results/latest/SUMMARY.md"
        break
      }
      Write-Host "!! push failed (attempt $i). Rebasing on remote and retrying..."
      git pull --rebase --autostash origin $Branch
      Start-Sleep -Seconds ([math]::Pow(2,$i))
    }
  }
} else {
  Write-Host "-- -NoPush set; results are in results/latest/ (not pushed)"
}

Write-Host "== done (pipeline rc=$runRc). See results/latest/SUMMARY.md"
exit $runRc
