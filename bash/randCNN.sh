CREATE_ROOT=false
TRAIN_MT_BASELINE=true
RUN_ATTACK=false
TEST=false


#echo $y
BashName=${0##*/}
#x=FileName
# main directory here
BashPath="$PWD"/$BashName
home_dir=/home/zsarwar/Projects/SparseDNNs/scripts

# Generic params
gpu=0
seed=42
attack="CW"
epsilon_list=(0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.02 0.022 0.024 0.026 0.028 0.03 0.032 0.034 0.036 0.038 0.04)
#epsilon_list=(0.032 0.034 0.036 0.038 0.04)
c_base=1.0
d_base=0.0

scale_factor=0
weight_repulsion="False"
sparseblock='None'
gaussian_dev=0.0

model='resnet18_randCNN'

#############################################################################
if [ "$model" = 'resnet18_randCNN' ] || [ "$model" = 'resnet18_sharded' ]
then    
    scale_factor=2
    weight_repulsion="True"
    sparseblock='SparsifyKernelGroups' # 'SparsifyKernelGroups'
fi

if [ "$model" = 'MLP-12' ] || [ "$model" = 'MLP-6' ] || [ "$model" = 'MLP-3' ]
then
    scale_factor=12
fi


if [ "$model" = 'resnet18_Gaussian' ]
then    
    gaussian_dev=0.015
fi

##############################################################################





# Setup the directory for an experiment
#############################################################################################

# MT Root parameters
base_dir='/net/scratch/zsarwar/SparseDNNs'
mt_dataset="cifar10"
mt_config="randCNN"
mt_classes=10
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
batch_size=512
lr=0.1
model_ema=False
epochs=2500
resume=True
pretrained=False
seed=42
eval_pretrained=False
num_classes=10
new_classifier=True
test_per_class=True
original_dataset=$mt_dataset
original_config=$mt_config
trainer_type="MT_Baseline"
mt_hash_config="${trainer_type}_${original_dataset}_${original_config}_${model}_pretrained-${pretrained}_lr-${lr}_batch_size-${batch_size}_epochs-${epochs}_eval-pretrained-${eval_pretrained}_scale-factor_${scale_factor}_sparseblock-${sparseblock}_gaussian-dev_${gaussian_dev}seed-${seed}"

if [ "$TRAIN_MT_BASELINE" = true ]
then
    cd ${home_dir}
    python3 train.py \
    --gpu=$gpu \
    --base_dir=$base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --epochs=$epochs \
    --arch=$model \
    --batch_size=$batch_size \
    --lr=$lr \
    --resume=$resume \
    --pretrained=$pretrained \
    --seed=$seed \
    --num_classes=$num_classes \
    --new_classifier=$new_classifier \
    --test_per_class=$test_per_class \
    --original_dataset=$original_dataset \
    --original_config=$original_config \
    --trainer_type=$trainer_type \
    --weight_repulsion=$weight_repulsion \
    --scale_factor=$scale_factor \
    --sparsefilter=$sparseblock \
    --gaussian_dev=$gaussian_dev

fi

#############################################################################################
# Attack parameters

RUN_ATTACK_TEST=true

# Attack parameters
original_dataset=cifar10
steps=2000
lr=0.01
batch_size=512
total_attack_samples_test=2560
attack_split='train'
integrated=False


if [ "$RUN_ATTACK" = true ]
then
    attack_split='test'

    if [ "$RUN_ATTACK_TEST" = true ]
    then
        for epsilon in "${epsilon_list[@]}"
        do       
            cd ${home_dir}
            python3 attack_bounded.py \
            --gpu=$gpu \
            --base_dir=$base_dir \
            --root_hash_config=$root_hash_config \
            --mt_hash_config=$mt_hash_config \
            --original_dataset=$original_dataset \
            --model=$model \
            --batch_size=$batch_size \
            --lr=$lr \
            --c=$c_base \
            --d=$d_base \
            --c_attack=$c_base \
            --d_attack=$d_base \
            --eps=$epsilon \
            --steps=$steps \
            --seed=$seed \
            --attack=$attack \
            --attack_split=$attack_split \
            --total_attack_samples=$total_attack_samples_test \
            --num_classes=$num_classes \
            --integrated=$integrated \
            --trainer_type=$trainer_type \
            --scale_factor=$scale_factor \
            --sparsefilter=$sparseblock
        done
    fi
fi

#############################################################################################
# Test adversarial samples on the model

TEST_ADVERSARIAL_TEST=true

test_type=adversarial
if [ "$TEST" = true ]
then
    attack_split='test'
    if [ "$TEST_ADVERSARIAL_TEST" = true ]
    then

        for epsilon in "${epsilon_list[@]}"
        do
            cd ${home_dir}
            python3 test.py \
            --gpu=$gpu \
            --base_dir=$base_dir \
            --root_hash_config=$root_hash_config \
            --mt_hash_config=$mt_hash_config \
            --epochs=$epochs \
            --arch=$model \
            --batch_size=$batch_size \
            --lr=$lr \
            --pretrained=$pretrained \
            --seed=$seed \
            --num_classes=$num_classes \
            --original_dataset=$original_dataset \
            --original_config=$original_config \
            --trainer_type=$trainer_type \
            --new_classifier=$new_classifier \
            --test_type=$test_type \
            --attack_split=$attack_split \
            --total_attack_samples=$total_attack_samples_test \
            --integrated=$integrated \
            --attack=$attack \
            --c=$c_base \
            --d=$d_base \
            --weight_repulsion=$weight_repulsion \
            --scale_factor=$scale_factor \
            --sparsefilter=$sparseblock \
            --gaussian_dev=$gaussian_dev \
            --eps=$epsilon
        done

    fi
fi
