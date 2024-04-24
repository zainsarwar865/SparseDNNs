#!/bin/bash
#SBATCH --partition=next-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=03:55:00
#SBATCH --output=/home/zsarwar/slurm/out/%j.%N.stdout
#SBATCH --error=/home/zsarwar/slurm/out/%j_.%N.stderr
#SBATCH --job-name=kernelCNN
#SBATCH --gres=gpu:h100:1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate "/home/zsarwar/.conda/envs/cnn"


./ResNet_gaussian.sh