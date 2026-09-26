# Download translation models for manga-ai (Windows)
# Usage:
#   hf auth login
#   .\scripts\download_models.ps1
#   .\scripts\download_models.ps1 -Model m2m100

param(
    [ValidateSet("all","m2m100","llama_en_fa","gemma_persian","gemma_gguf","opus")]
    [string]$Model = "all"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

pip install -q huggingface_hub
python scripts/download_models.py --model $Model
