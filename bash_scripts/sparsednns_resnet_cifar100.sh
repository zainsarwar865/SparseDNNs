CREATE_ROOT=true
TRAIN_MT_BASELINE=true
RUN_ATTACK=false
TEST_ADVERSARIAL=false
FEATURE_EXTRACTION_BENIGN=false
FEATURE_EXTRACTION_ADVERSARIAL=false
TRAIN_RBF=false

TEST_MT_INTEGRATED_PREATTACK=false
RUN_ATTACK_INTEGRATED=false
TEST_INTEGRATED_ADVERSARIAL=false

#echo $y
BashName=${0##*/}
#x=FileName
# main directory here
BashPath="$PWD"/$BashName
home_dir=/home/zsarwar/Projects/SparseDNNs

# Generic params
gpu=0
seed=42
attack="CW"
detector_type="Quantized" # Regular

C=0.99
gamma=0.99


if [ "$detector_type" = 'Quantized' ]
then
    C=0.9
    gamma=0.05

fi


# Setup the directory for an experiment
#############################################################################################

# MT Root parameters
base_dir='/bigstor/zsarwar/SparseDNNs'
mt_dataset="cifar100"
mt_config="full"
mt_classes=100
# root_config --> subset
# Hash configs
root_hash_config="MT_${mt_dataset}_${mt_config}_${mt_classes}"
if [ "$CREATE_ROOT" = true ]
then
    cd ${home_dir}
    python3 utils/create_directories.py \
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
batch_size=256
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
num_eval_epochs=10
resume=''
pretrained=False
freeze_layers=False
seed=42
eval_pretrained=False
num_classes=100
new_classifier=True
external_augmentation=False
test_per_class=True
original_dataset=$mt_dataset
original_config=$mt_config
model='resnet18'
trainer_type="MT_Baseline"
mt_hash_config="${trainer_type}_${original_dataset}_${original_config}_${model}_pretrained-${pretrained}_freeze-layers-${freeze_layers}_lr-${lr}_batch_size-${batch_size}_lr-warmup-epochs-${lr_warmup_epochs}_lr-warmup-decay-${lr_warmup_decay}_label-smoothing-${label_smoothing}_mixup-alpha-${mixum_alpha}_cutmix_alpha-${cutmix_alpha}_random-erasing-${random_erasing}_model-ema-${model_ema}_weight_decay-${weight_decay}_epochs-${epochs}_eval-pretrained-${eval_pretrained}_seed-${seed}"

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
original_dataset=cifar100
c=0.06
steps=200
lr=0.01
batch_size=512
total_attack_samples_train=5120
total_attack_samples_test=5120
attack_split='train'
integrated=False


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
        --detector_type=$detector_type \
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
        --detector_type=$detector_type \
        --total_attack_samples=$total_attack_samples_test \
        --num_classes=$num_classes \
        --integrated=$integrated \
        --trainer_type=$trainer_type
    fi
fi



#############################################################################################
# Test adversarial samples on the model

TEST_MT_ADVERSARIAL_TRAIN=false
TEST_MT_ADVERSARIAL_TEST=true
test_type=adversarial
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
        --test_type=$test_type \
        --attack_split=$attack_split \
        --detector_type=$detector_type \
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
        --test_type=$test_type \
        --attack_split=$attack_split \
        --detector_type=$detector_type \
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
        --detector_type=$detector_type \
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
        --detector_type=$detector_type \
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
        --detector_type=$detector_type \
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
        --detector_type=$detector_type \
        --total_attack_samples=$total_attack_samples_test \
        --integrated=$integrated \
        --attack=$attack
    fi
fi


#############################################################################################
# Train RBF Kernel
train='True'
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
    --detector_type=$detector_type \
    --total_attack_samples=$total_attack_samples_test \
    --total_train_samples=$total_attack_samples_train \
    --train=$train \
    --C=$C \
    --gamma=$gamma

fi








#############################################################################################
# Start integrated attack

#############################################################################################
# Attack parameters

original_dataset=cifar100
c=0.06
d=2
steps=500
lr=0.01
batch_size=256
total_attack_samples_test=5120


#############################################################################################
# Get baseline performance of samples to be attacked
total_attack_samples=$total_attack_samples_train
integrated=False
test_type=adversarial
attack_split='test'
extract_split=$attack_split
extract_type=$test_type
train=False
    if [ "$TEST_MT_INTEGRATED_PREATTACK" = true ]
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
        --test_type=$test_type \
        --attack_split=$attack_split \
        --detector_type=$detector_type \
        --total_attack_samples=$total_attack_samples_test \
        --integrated=$integrated \
        --attack=$attack
    
    
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
        --detector_type=$detector_type \
        --total_attack_samples=$total_attack_samples_test \
        --integrated=$integrated \
        --attack=$attack

        # Test RBF Kernel
        train='False'
        cd ${home_dir}
        python3 rbf.py \
        --gpu=$gpu \
        --base_dir=$base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --seed=$seed \
        --attack=$attack \
        --trainer_type=$trainer_type \
        --total_attack_samples=$total_attack_samples_test \
        --total_train_samples=$total_attack_samples \
        --train=$train \
        --test_type=$test_type \
        --integrated=$integrated \
        --detector_type=$detector_type

    fi


#############################################################################################
# Start integrated attack

# Attack parameters


integrated=True
test_type=adversarial
extract_type=$test_type

if [ "$RUN_ATTACK_INTEGRATED" = true ]
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
    --d=$d \
    --steps=$steps \
    --seed=$seed \
    --attack=$attack \
    --attack_split=$attack_split \
    --detector_type=$detector_type \
    --total_attack_samples=$total_attack_samples_test \
    --num_classes=$num_classes \
    --integrated=$integrated \
    --total_train_samples=$total_attack_samples \
    --trainer_type=$trainer_type

fi

#############################################################################################
# Test adversarial samples

if [ "$TEST_INTEGRATED_ADVERSARIAL" = true ]
then
    attack_split='test'
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
    --test_type=$test_type \
    --attack_split=$attack_split \
    --detector_type=$detector_type \
    --total_attack_samples=$total_attack_samples_test \
    --integrated=$integrated \
    --attack=$attack

    # Feature extraction from adversarial samples
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
    --detector_type=$detector_type \
    --total_attack_samples=$total_attack_samples_test \
    --integrated=$integrated \
    --attack=$attack

    # Test RBF
    cd ${home_dir}
    python3 rbf.py \
    --gpu=$gpu \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --seed=$seed \
    --attack=$attack \
    --trainer_type=$trainer_type \
    --total_attack_samples=$total_attack_samples_test \
    --total_train_samples=$total_attack_samples \
    --train=$train \
    --test_type=$test_type \
    --integrated=$integrated \
    --detector_type=$detector_type
fi
#############################################################################################
# THE END