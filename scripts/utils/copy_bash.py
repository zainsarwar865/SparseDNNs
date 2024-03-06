import os
import argparse
import hashlib
import shutil 

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--base_dir')
parser.add_argument('--root_hash_config')
parser.add_argument('--bash_script_config', type=str)
args = parser.parse_args()
root_config = args.root_hash_config
root_config_hash = (hashlib.md5(root_config.encode('UTF-8')))
mt_config = root_config + "_" + root_config_hash.hexdigest()

mt_root_directory = os.path.join(args.base_dir, mt_config)
bashname = args.bash_script_config.split("/")[-1]
bash_copy_path = os.path.join(mt_root_directory, bashname)
shutil.copyfile(args.bash_script_config, bash_copy_path)
