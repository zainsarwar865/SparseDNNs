CREATE_ROOT=false
TRAIN_MT_BASELINE=false
RUN_ATTACK=false
TEST_ADVERSARIAL=false
FEATURE_EXTRACTION_BENIGN=false
FEATURE_EXTRACTION_ADVERSARIAL=false
TRAIN_RBF=false



RUN_ATTACK_INTEGRATED=false
TEST_INTEGRATED_ADVERSARIAL=false
FEATURE_EXTRACTION_INTEGRATED_ADVERSARIAL=false
TEST_ATTACK_INTEGRATED=true


#echo $y
BashName=${0##*/}
#x=FileName
# main directory here
BashPath="$PWD"/$BashName
home_dir=/home/zsarwar/Projects/SparseDNNs

# Generic params
gpu=1
seed=42
attack="CW"

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
# Copy bash script to root dir
cd ${home_dir}
    python3 utils/copy_bash.py \
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
# Attack parameters

RUN_ATTACK_TRAIN=true
RUN_ATTACK_TEST=true

# Attack parameters
original_dataset=cifar10
c=0.02
steps=200
lr=0.01
batch_size=512
total_attack_samples_train=5120
attack_split='train'
integrated=False

total_attack_samples_test=5120

if [ "$RUN_ATTACK" = true ]
then

    if [ "$RUN_ATTACK_TRAIN" = true ]
    then
        cd ${home_dir}
        python3 attack.py \
        --gpu=$gpu \
        --base_dir=$base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --original_dataset=$original_dataset \
        --model=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --c=$c \
        --steps=$steps \
        --seed=$seed \
        --attack=$attack \
        --attack_split=$attack_split \
        --total_attack_samples=$total_attack_samples_train \
        --num_classes=$num_classes \
        --integrated=$integrated \
        --trainer_type=$trainer_type
    fi


attack_split='test'


    if [ "$RUN_ATTACK_TEST" = true ]
    then
        
        cd ${home_dir}
        python3 attack.py \
        --gpu=$gpu \
        --base_dir=$base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --original_dataset=$original_dataset \
        --model=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --c=$c \
        --steps=$steps \
        --seed=$seed \
        --attack=$attack \
        --attack_split=$attack_split \
        --total_attack_samples=$total_attack_samples_test \
        --num_classes=$num_classes \
        --integrated=$integrated \
        --trainer_type=$trainer_type
    fi
fi

#############################################################################################
# Test adversarial samples

TEST_MT_ADVERSARIAL_TRAIN=true
TEST_MT_ADVERSARIAL_TEST=true

test_adversarial=True


if [ "$TEST_ADVERSARIAL" = true ]
then

    attack_split='train'
    if [ "$TEST_MT_ADVERSARIAL_TRAIN" = true ]
    then
        cd ${home_dir}
        python3 test.py \
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
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --trainer_type=$trainer_type \
        --new_classifier=$new_classifier \
        --test_adversarial=$test_adversarial \
        --attack_split=$attack_split \
        --total_attack_samples=$total_attack_samples_train \
        --integrated=$integrated \
        --attack=$attack

    fi

    attack_split='test'
    if [ "$TEST_MT_ADVERSARIAL_TEST" = true ]
    then
        cd ${home_dir}
        python3 test.py \
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
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --trainer_type=$trainer_type \
        --new_classifier=$new_classifier \
        --test_adversarial=$test_adversarial \
        --attack_split=$attack_split \
        --total_attack_samples=$total_attack_samples_test \
        --integrated=$integrated \
        --attack=$attack
    fi

fi
#############################################################################################
# Feature extraction from benign samples


if [ "$FEATURE_EXTRACTION_BENIGN" = true ]
then

    FEATURE_EXTRACTION_BENIGN_TRAIN=true
    FEATURE_EXTRACTION_BENIGN_TEST=true

    extract_type=benign
    extract_split=train

    if [ "$FEATURE_EXTRACTION_BENIGN_TRAIN" = true ]
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
        --trainer_type=$trainer_type \
        --extract_type=$extract_type \
        --extract_split=$extract_split \
        --total_attack_samples=$total_attack_samples_train \
        --integrated=$integrated \
        --attack=$attack

    fi


    extract_split=test

    if [ "$FEATURE_EXTRACTION_BENIGN_TEST" = true ]
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
        --trainer_type=$trainer_type \
        --extract_type=$extract_type \
        --extract_split=$extract_split \
        --total_attack_samples=$total_attack_samples_test \
        --integrated=$integrated \
        --attack=$attack

    fi


fi

#############################################################################################
# Feature extraction from adversarial samples

if [ "$FEATURE_EXTRACTION_ADVERSARIAL" = true ]
then

    FEATURE_EXTRACTION_ADVERSARIAL_TRAIN=true
    FEATURE_EXTRACTION_ADVERSARIAL_TEST=true

    extract_type=adversarial
    extract_split=train

    if [ "$FEATURE_EXTRACTION_ADVERSARIAL_TRAIN" = true ]
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
        --trainer_type=$trainer_type \
        --extract_type=$extract_type \
        --extract_split=$extract_split \
        --total_attack_samples=$total_attack_samples_train \
        --integrated=$integrated \
        --attack=$attack
    fi

    extract_split=test

    if [ "$FEATURE_EXTRACTION_ADVERSARIAL_TEST" = true ]
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
        --trainer_type=$trainer_type \
        --extract_type=$extract_type \
        --extract_split=$extract_split \
        --total_attack_samples=$total_attack_samples_test \
        --integrated=$integrated \
        --attack=$attack
    fi
fi


#############################################################################################
# Train RBF Kernel

if [ "$TRAIN_RBF" = true ]
then
    cd ${home_dir}
    python3 rbf.py \
    --gpu=$gpu \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --seed=$seed \
    --attack=$attack \
    --trainer_type=$trainer_type \
    --integrated=$integrated \
    --total_attack_samples=$total_attack_samples_test
fi










#############################################################################################
# Run attack with rbf loss integrated


#############################################################################################
# Attack parameters
RUN_ATTACK_INTEGRATED_TRAIN=true
RUN_ATTACK_INTEGRATED_TEST=false

# Attack parameters

total_attack_samples=$total_attack_samples_train
integrated=True

original_dataset=cifar10
c=0.02
steps=200
lr=0.01
batch_size=256
total_attack_samples_train=256
attack_split='train'

total_attack_samples_test=256

if [ "$RUN_ATTACK_INTEGRATED" = true ]
then
    if [ "$RUN_ATTACK_INTEGRATED_TRAIN" = true ]
    then
        cd ${home_dir}
        python3 attack.py \
        --gpu=$gpu \
        --base_dir=$base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --original_dataset=$original_dataset \
        --model=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --c=$c \
        --steps=$steps \
        --seed=$seed \
        --attack=$attack \
        --attack_split=$attack_split \
        --total_attack_samples=$total_attack_samples_train \
        --num_classes=$num_classes \
        --integrated=$integrated \
        --total_train_samples=$total_attack_samples \
        --trainer_type=$trainer_type
    fi


fi

# TODO
# DO TEST SPLIT HERE


#############################################################################################
# Test adversarial samples

TEST_INTEGRATED_ADVERSARIAL_TRAIN=true
TEST_INTEGRATED_ADVERSARIAL_TEST=false
test_adversarial=True

if [ "$TEST_INTEGRATED_ADVERSARIAL" = true ]
then

    attack_split='train'
    if [ "$TEST_INTEGRATED_ADVERSARIAL_TRAIN" = true ]
    then
        cd ${home_dir}
        python3 test.py \
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
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --trainer_type=$trainer_type \
        --new_classifier=$new_classifier \
        --test_adversarial=$test_adversarial \
        --attack_split=$attack_split \
        --total_attack_samples=$total_attack_samples_train \
        --integrated=$integrated \
        --attack=$attack

    fi

 
fi
#############################################################################################


#############################################################################################
# Feature extraction from adversarial samples
FEATURE_EXTRACTION_INTEGRATED_ADVERSARIAL_TRAIN=true
FEATURE_EXTRACTION_INTEGRATED_ADVERSARIAL_TEST=false
extract_type=adversarial
extract_split=train

if [ "$FEATURE_EXTRACTION_INTEGRATED_ADVERSARIAL" = true ]
then


   
    if [ "$FEATURE_EXTRACTION_INTEGRATED_ADVERSARIAL_TRAIN" = true ]
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
        --trainer_type=$trainer_type \
        --extract_type=$extract_type \
        --extract_split=$extract_split \
        --total_attack_samples=$total_attack_samples_train \
        --integrated=$integrated \
        --attack=$attack
    fi
fi


#############################################################################################
# Test RBF Kernel
if [ "$TEST_ATTACK_INTEGRATED" = true ]
then
    cd ${home_dir}
    python3 rbf_test.py \
    --gpu=$gpu \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --seed=$seed \
    --attack=$attack \
    --attack_split=$extract_split \
    --trainer_type=$trainer_type \
    --total_attack_samples=$total_attack_samples_train \
    --total_train_samples=$total_attack_samples
fi
#############################################################################################