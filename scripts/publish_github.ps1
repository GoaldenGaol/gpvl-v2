# Publish GPVL v2 to GitHub (GoaldenGaol/gpvl-v2)
# Requires: GitHub CLI authenticated (gh auth login)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$Gh = (Get-Command gh -ErrorAction SilentlyContinue)?.Source
if (-not $Gh) {
    $candidates = @(
        "C:\Program Files\GitHub CLI\gh.exe",
        "C:\Program Files (x86)\GitHub CLI\gh.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $Gh = $candidate
            break
        }
    }
}
if (-not $Gh) {
    throw "GitHub CLI (gh) not found. Install from https://cli.github.com/"
}

Set-Location $RepoRoot

& $Gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Not authenticated. Run: gh auth login"
}

$repo = "GoaldenGaol/gpvl-v2"
& $Gh repo view $repo 2>$null | Out-Null
$exists = $LASTEXITCODE -eq 0

if (-not $exists) {
    Write-Host "Creating public repository $repo ..."
    & $Gh repo create $repo `
        --public `
        --description "GPVL v2 — General Physics of Volition: cooperative equilibria, dim4 dynamics, and VPDE modeling" `
        --homepage "https://github.com/GoaldenGaol/science-of-volition"
    if ($LASTEXITCODE -ne 0) { throw "gh repo create failed" }
}

if (-not (git remote get-url origin 2>$null)) {
    git remote add origin "https://github.com/$repo.git"
}

$branch = git branch --show-current
Write-Host "Pushing branch $branch to origin ..."
git push -u origin $branch

& $Gh repo edit $repo `
    --add-topic volition `
    --add-topic dim4 `
    --add-topic cooperative-equilibria `
    --add-topic vpde `
    --add-topic python `
    --add-topic reproducible-research

Write-Host "Published: https://github.com/$repo"