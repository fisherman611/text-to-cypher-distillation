param(
    [string]$Benchmark = "Cypherbench",
    [string]$Db = "full",
    [string]$Model = "Qwen/Qwen3-0.6B",
    [string]$DataSource = "hf",
    [string]$HfDatasetRepo = "fisherman611/text_to_cypher_distillation",
    [string]$HfDatasetRevision = "main",
    [string]$CkptPath = "",
    [string]$CkptRevision = "",
    [string]$Device = "auto",
    [int]$MaxLength = 1024,
    [int]$BatchSize = 1,
    [double]$Temperature = 0.5,
    [double]$TopP = 0.95,
    [int]$TopK = 0,
    [string]$Limit = ""
)

$argsList = @(
    "infer.py",
    "--benchmark", $Benchmark,
    "--data_source", $DataSource,
    "--hf_dataset_repo", $HfDatasetRepo,
    "--hf_dataset_revision", $HfDatasetRevision,
    "--db", $Db,
    "--model", $Model,
    "--device", $Device,
    "--max-length", "$MaxLength",
    "--batch-size", "$BatchSize",
    "--temperature", "$Temperature",
    "--top-p", "$TopP",
    "--top-k", "$TopK"
)

if ($CkptPath) {
    $argsList += @("--ckpt_path", $CkptPath)
}

if ($CkptRevision) {
    $argsList += @("--ckpt_revision", $CkptRevision)
}

if ($Limit) {
    $argsList += @("--limit", $Limit)
}

Write-Host ("Running: python " + ($argsList -join " "))
python @argsList
