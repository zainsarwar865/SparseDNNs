# Five main components of the Enola Simulator
CREATE_ROOT=false
TRAIN_MT_BASELINE=false
FEATURE_EXTRACTION=false


#echo $y
BashName=${0##*/}
#x=FileName
# main directory here
BashPath="$PWD"/$BashName
home_dir=/home/zsarwar/Projects/SparseDNNs

# Generic params
gpu=0
seed=42


# Setup the directory for an experiment
#############################################################################################

# MT Root parameters
base_dir='/bigstor/zsarwar/SparseDNNs'
mt_dataset="CIFAR10"
mt_config="full"
mt_classes=10
# root_config --> subset
# Hash configs
root_hash_config="MT_${mt_dataset}_${mt_config}_${mt_classes}"
if [ "$CREATE_ROOT" = true ]
then
    cd ${home_dir}
    python3 create_directories.py \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    
fi

#############################################################################################



#############################################################################################
# Copy bash script to root dir
cd ${home_dir}
    python3 copy_bash.py \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --bash_script_config=$BashPath


#############################################################################################

# Train MT's baseline model
#############################################################################################

# MT train parameters
batch_size=128
lr=0.1
weight_decay=0.0001
lr_warmup_epochs=2
lr_warmup_decay=0.01
label_smoothing=0.1
mixup_alpha=0.2
cutmix_alpha=0.2
random_erasing=0.1
model_ema=False
epochs=1000
num_eval_epochs=1
resume=''
pretrained=False
freeze_layers=False
seed=42
eval_pretrained=False
num_classes=10
new_classifier=True
external_augmentation=False
test_per_class=True
original_dataset=$mt_dataset
original_config=$mt_config
model='resnet18'
trainer_type="MT_Baseline"
mt_hash_config="${trainer_type}_${original_dataset}_${original_config}_${model}_pretrained-${pretrained}_freeze-layers-${freeze_layers}_lr-${lr}_batch_size-${batch_size}_lr-warmup-epochs-${lr_warmup_epochs}_lr-warmup-decay-${lr_warmup_decay}_label-smoothing-${label_smoothing}_mixup-alpha-${mixum_alpha}_cutmix_alpha-${cutmix_alpha}_random-erasing-${random_erasing}_model-ema-${model_ema}_weight_decay-${weight_decay}_epochs-${epochs}_eval-pretrained-${eval_pretrained}_seed-{$seed}"

if [ "$TRAIN_MT_BASELINE" = true ]
then
    cd ${home_dir}
    python3 train.py \
    --gpu=$gpu \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --epochs=$epochs \
    --num_eval_epochs=$num_eval_epochs \
    --arch=$model \
    --batch_size=$batch_size \
    --lr=$lr \
    --weight_decay=$weight_decay \
    --lr_warmup_epochs=$lr_warmup_epochs \
    --lr_warmup_decay=$lr_warmup_decay \
    --label_smoothing=$label_smoothing \
    --mixup_alpha=$mixup_alpha \
    --cutmix_alpha=$cutmix_alpha \
    --random_erasing=$random_erasing \
    --model_ema=False \
    --resume=$resume \
    --pretrained=$pretrained \
    --freeze_layers=$freeze_layers \
    --seed=$seed \
    --num_classes=$num_classes \
    --new_classifier=$new_classifier \
    --test_per_class=$test_per_class \
    --original_dataset=$original_dataset \
    --original_config=$original_config \
    --trainer_type=$trainer_type
fi

#############################################################################################

#############################################################################################
# Feature extraction

if [ "$FEATURE_EXTRACTION" = true ]
then
    cd ${home_dir}
    python3 feature_extraction.py \
    --gpu=$gpu \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --epochs=$epochs \
    --num_eval_epochs=$num_eval_epochs \
    --arch=$model \
    --batch_size=$batch_size \
    --lr=$lr \
    --weight_decay=$weight_decay \
    --lr_warmup_epochs=$lr_warmup_epochs \
    --lr_warmup_decay=$lr_warmup_decay \
    --label_smoothing=$label_smoothing \
    --mixup_alpha=$mixup_alpha \
    --cutmix_alpha=$cutmix_alpha \
    --random_erasing=$random_erasing \
    --model_ema=False \
    --resume=$resume \
    --pretrained=$pretrained \
    --freeze_layers=$freeze_layers \
    --seed=$seed \
    --num_classes=$num_classes \
    --new_classifier=$new_classifier \
    --test_per_class=$test_per_class \
    --original_dataset=$original_dataset \
    --original_config=$original_config \
    --trainer_type=$trainer_type
fi
#############################################################################################