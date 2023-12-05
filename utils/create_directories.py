import os
import argparse
import hashlib

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--base_dir', default='/bigstor/zsarwar/SparseDNNs/')
parser.add_argument('--root_hash_config', default='')

args = parser.parse_args()
root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()

mt_root_directory = os.path.join(args.base_dir, mt_config)
mt_baseline_dir = "MT_Baseline"
datasets_dir = "Datasets"
attack_dir = "Adversarial_Datasets"
benign_dir = "Benign_Datasets"
metrics_dir = "Metrics"

# Create root
if not os.path.exists(mt_root_directory):
    os.makedirs(mt_root_directory)
else:
    print("MT's root directory already exists")

datasets_path = os.path.join(mt_root_directory, datasets_dir)
if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

attack_path = os.path.join(mt_root_directory, attack_dir)
if not os.path.exists(attack_path):
    os.makedirs(attack_path)

benign_path = os.path.join(mt_root_directory, benign_dir)
if not os.path.exists(benign_path):
    os.makedirs(benign_path)


print("Created root directory")