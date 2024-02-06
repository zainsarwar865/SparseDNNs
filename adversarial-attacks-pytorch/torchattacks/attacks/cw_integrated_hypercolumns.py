import torch
import torch.nn as nn
import torch.optim as optim
import sys
from ..attack import Attack
import numpy as np


torch.manual_seed(42)


class CW_MLP(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """
    def __init__(self, model, mlp, mask, c=1, d=0.001,kappa=0, steps=50, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.d = d
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]
        self.mlp = mlp
        self.sample_dims = 200
        self.mask = mask

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)
        flipped_mask = torch.zeros(images.shape[0], dtype=torch.bool).to(self.device)
        mlp_mask = torch.zeros(images.shape[0], dtype=torch.bool).to(self.device)
        for step in range(self.steps):
            print("On step : ", step)
            # Get adversarial images
            adv_images = self.tanh_space(w)
            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs, relu_feats = self.get_logits(adv_images)
            # Construct hypercolumn features
            hyper_feats = self.hypercolumn_feature_extractor(relu_feats, adv_images.shape[0])

            # MLP preds and image level classification
            true_labels = torch.ones(hyper_feats.shape[0]).to(self.device)
            mlp_scores = self.mlp(hyper_feats)
            pred_labels = torch.argmax(mlp_scores, dim=1)
            ####################################
            benign_votes = torch.tensor([0 for i in range(len(true_labels) // self.sample_dims)]).to(self.device)
            adversarial_votes = torch.tensor([0 for i in range(len(true_labels) // self.sample_dims)]).to(self.device)
            for idx, i in enumerate(range(0, len(true_labels), self.sample_dims)):
                curr_pred = pred_labels[i:i+self.sample_dims]
                adv_votes = torch.sum(curr_pred == 1)
                ben_votes = torch.sum(curr_pred == 0)
                adversarial_votes[idx] = adv_votes
                benign_votes[idx] = ben_votes

            mlp_outputs = torch.stack((benign_votes, adversarial_votes), dim=0)
            
            image_preds = torch.argmax(mlp_outputs, dim=0)            
            y_images = true_labels[0:len(image_preds)]
            # Img acc
            img_acc = sum(torch.eq(image_preds, y_images).to(dtype=int)) / len(image_preds)
            print("Image level accuracy for MLP detector is : ", img_acc)
            ####################################
            # mlp loss            
            mlp_pred_cw = torch.clamp(mlp_scores[:, 1] - mlp_scores[:, 0], 0)
            red_indices = torch.where(image_preds == 0)[0]
            print("Pos results are: ", red_indices.shape)
            print("Batch size is ", image_preds.shape)
            mlp_loss = mlp_pred_cw.sum()
            if step > 10:
                mlp_t_mask = image_preds == 0
                mlp_mask = torch.logical_or(mlp_t_mask, mlp_mask)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()
            cost = L2_loss + self.c * f_loss
            #cost = L2_loss + self.c * f_loss + self.d * mlp_loss 
            print("L2_loss: ",  L2_loss.item())
            print("mlp_loss: ",  mlp_loss.item())
            print("f_loss: ",  f_loss.item())
            print("--------------------")
            print("cost: ",  cost.item())
            print("--------------------")
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                condition = (pre == target_labels).float()
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()
                #print(mlp_labels)
                mlp_condition = image_preds == 0
                condition = condition * mlp_condition
            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            
            mask = condition * (best_L2 > current_L2.detach())
            flipped_mask = torch.logical_or(mask, flipped_mask)

            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            """
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    best_adv_images = best_adv_images.detach().cpu()
                    adv_images = adv_images.detach().cpu()
                    #return best_adv_images, images.detach().cpu()
                    return adv_images, images.detach().cpu()
                prev_cost = cost.item()
            """
        # Only return successful adversarial samples
        flipped_indices = torch.where(flipped_mask == True)[0]
        #best_adv_images = best_adv_images[flipped_indices]
        #images = images[flipped_indices]
        #sub_labels = labels.detach()[flipped_indices].cpu()
        flipped_indices = flipped_indices.detach().cpu()
        best_adv_images = best_adv_images.detach().cpu()
        adv_images = adv_images.detach().cpu()       
        return best_adv_images, images.detach().cpu(), flipped_indices

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
    

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]
        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)
    
    def hypercolumn_feature_extractor(self, relu_feats, bsize):
        x6, x4, x3, x1, x0 = relu_feats
        # Process and Concat features
        x6 = x6.unsqueeze(-1).unsqueeze(-1)
        up_size = (32, 32)
        upsample_layer = torch.nn.Upsample(size=up_size, mode='bilinear')
        x6 = upsample_layer(x6)
        x4 = upsample_layer(x4)
        x3 = upsample_layer(x3)
        x1 = upsample_layer(x1)
        x0 = upsample_layer(x0)
        # Concat across filter dimension and reshape
        features = torch.concat((x6, x4, x3, x1, x0), dim=1)
        features = features.flatten(2,3)
        features = features.transpose(1,2)
        mask = self.mask[0:bsize]
        subselected_features = features[mask]
        features = subselected_features.view(features.shape[0], self.sample_dims, features.shape[2])
        features = features.flatten(0,1)
        
        return features