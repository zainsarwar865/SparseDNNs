import torchattacks

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

# GO-GO-GO!
normalize  =  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    

# Undo normalization for now
transform_test = transforms.Compose([
    transforms.ToTensor(), 
    ])


valset = torchvision.datasets.CIFAR10(root='/bigstor/zsarwar/CIFAR10', train=False,
                                    download=False, transform=transform_test
                                        )
                                    
dataloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=2)
   
model = resnet18()
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
loc = 'cuda:{}'.format(1)
# Load best checkpoint
best_ckpt_path = "/bigstor/zsarwar/SparseDNNs/MT_CIFAR10_full_10_d5f3f545a0883adb9c8f98e2a6ba4ac7/MT_Baseline_d2a45a4dd02a5e037e5954b82387e666/Checkpoints/model_best.pth.tar"
checkpoint = torch.load(best_ckpt_path, map_location=loc)
model.load_state_dict(checkpoint['state_dict'])

mean = (0.4914, 0.4822, 0.4465)
std = (0.247, 0.243, 0.261)

images, labels = next(iter(dataloader))
atk = torchattacks.CW(model, c=1, steps=1000, lr=0.01)
# If inputs were normalized, then
atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
adv_images = atk(images, labels)
print(adv_images)