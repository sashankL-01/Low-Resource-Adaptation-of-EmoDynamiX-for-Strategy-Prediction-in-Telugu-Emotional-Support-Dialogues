# PowerShell training script for Telugu heterogeneous graph strategy prediction
param(
    [string]$Model="xlmr-hg-telugu",
    [string]$Dataset="esconv-telugu",
    [int]$TotalSteps=5000,
    [int]$Epochs=10,
    [int]$BatchSize=8,
    [int]$SaveSteps=500,
    [int]$EvalSteps=500,
    [float]$LR=2e-5,
    [float]$WeightDecay=1e-3,
    [int]$Warmup=500,
    [int]$ExcludeOthers=0,
    [int]$ErcMixed=1,
    [float]$ErcTemperature=0.5,
    [int]$HgDim=512,
    [string]$TeluguErcPath="telugu_erc_xlmroberta_trained_v2"
)

python main.py --mode train --model $Model --dataset $Dataset --total_steps $TotalSteps --total_epochs $Epochs --batch_size $BatchSize --save_steps $SaveSteps --eval_steps $EvalSteps --lr $LR --weight_decay $WeightDecay --warmup $Warmup --exclude_others $ExcludeOthers --erc_mixed $ErcMixed --erc_temperature $ErcTemperature --hg_dim $HgDim --telugu_erc_path $TeluguErcPath
