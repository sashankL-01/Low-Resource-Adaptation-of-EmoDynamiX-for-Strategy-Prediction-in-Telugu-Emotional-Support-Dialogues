# PowerShell inference script: evaluate a saved checkpoint on Telugu dataset
param(
    [int]$CheckpointStep,
    [string]$Model="xlmr-hg-telugu",
    [string]$Dataset="esconv-telugu",
    [int]$BatchSize=8,
    [string]$TeluguErcPath="telugu_erc_xlmroberta_trained_v2"
)

python main.py --mode test --load_checkpoint $CheckpointStep --model $Model --dataset $Dataset --batch_size $BatchSize --telugu_erc_path $TeluguErcPath
