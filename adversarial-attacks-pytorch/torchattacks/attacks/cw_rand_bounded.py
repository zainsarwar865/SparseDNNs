import torch
import torch.nn as nn
import torch.optim as optim
from ..attack_rand import Attack

class CW(Attack):
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

    def __init__(self, model, c=1, kappa=0, steps=50, lr=0.01, eps=0.015):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]
        self.eps = eps
        #self.set_mode_targeted_random()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        print("C value is : ", self.c)
        print(f"Eps is {self.eps}")
        images = images.clone().detach().to(self.device)
        #print("Inside forward", images[0])
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

        # For tracking moving average of successes/failures
        batch_size = images.shape[0]
        cache_size =  50
        success_thres = 0.5

        moving_results = torch.zeros(size=(batch_size, cache_size)).to(device=self.device)
        moving_results[:] = torch.nan
        moving_avg = torch.zeros(size=(batch_size,)).to(device=self.device)
        best_moving_avg = torch.negative(torch.ones(size=(batch_size,))).to(device=self.device)

        for step in range(self.steps):
            # Get adversarial images
            print("On step : ", step)
            t_adv_images = self.tanh_space(w)
            adv_images = (self.eps*(t_adv_images - images)) + images

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()
            cost = f_loss
            print("L2_loss: ",  L2_loss.item())
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
                #print(torch.sum(condition), condition.shape)
            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            
            #mask = condition * (best_L2 > current_L2.detach())
            mask = condition
            flipped_mask = torch.logical_or(mask, flipped_mask)
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            # Update running average values
            binary_success = mask
            cache_idx = step % cache_size
            moving_results[:, cache_idx] = binary_success
            moving_avg = torch.nanmean(moving_results, dim=1)
            mask_mov_avg = moving_avg > best_moving_avg
            mask = torch.logical_and(mask_mov_avg, mask)
            best_moving_avg[mask] = moving_avg[mask]
            mask = mask.float()
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images



        # Only return successful adversarial samples
        flipped_indices = torch.where(flipped_mask == True)[0]
        flipped_indices = flipped_indices.detach().cpu()
        best_adv_images = best_adv_images.detach().cpu()
        best_moving_avg = best_moving_avg.detach().cpu()
        return best_adv_images, images.cpu(), flipped_indices, best_moving_avg

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
