from collections import defaultdict
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision.models.resnet import Bottleneck
import tqdm


class StatsManager:
    def __init__(self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str,
        use_gradient_checkpoint: bool = False
    ):
        self.model = model.to(device)
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.data_loader = data_loader
        self.device = device
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self._setup_hooks()
        self.gradients_mean = defaultdict(float)
        self.gradients_2nd_moment = defaultdict(float)
    
    def collect_stats(self) -> None:
        self.n_batch = 0
        for x, y_true, _ in (pbar := tqdm.tqdm(self.data_loader)):
            self.n_batch += 1
            # self.relu_invocation_counter = defaultdict(int)
            self.max_forward_update = 0
            self.max_backward_update = 0
            self.max_grad_update = 0
            
            if self.use_gradient_checkpoint:
                x.requires_grad = True
                y_pred = checkpoint(self.model, x.to(self.device))
            else:
                y_pred = self.model(x.to(self.device))
            loss = F.cross_entropy(y_pred, y_true.to(self.device))
            loss.backward()

            self._save_gradients()
            self.model.zero_grad()

            pbar.set_description(f'max update|fwd{self.max_backward_update:.0e}|bwk{self.max_backward_update:.0e}|grad{self.max_grad_update:.0e}')
    
    def save_stats(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        attrs = [
            'forward_intermediates_mean', 
            'forward_intermediates_2nd_moment', 
            'backward_intermediates_mean', 
            'backward_intermediates_2nd_moment',
            'gradients_mean',
            'gradients_2nd_moment'
        ]

        for attr in attrs:
            with open(os.path.join(path, f'{attr}.pickle'), 'wb') as fp:
                pickle.dump(
                    {
                        k: v.cpu().numpy()
                        for k, v in getattr(self, attr).items()
                    }, fp)

    def _setup_hooks(self):
        self.forward_intermediates_mean = defaultdict(float)
        self.forward_intermediates_2nd_moment = defaultdict(float)
        self.activation_probability = defaultdict(float)
        
        self.backward_intermediates_mean = defaultdict(float)
        self.backward_intermediates_2nd_moment = defaultdict(float)

        for name, module in self.model.named_modules():
            if isinstance(module, Bottleneck) or name == 'fc':
                def forward_hook(module, input, output, name=name):
                    output_detached = output.detach()
                    mean = output_detached.mean(0)
                    second_moment = (output_detached ** 2).mean(0)
                    # if isinstance(module, nn.ReLU):
                    #     self.relu_invocation_counter[name] += 1
                    #     name += str(self.relu_invocation_counter[name])

                    mean_update = (
                        mean - self.forward_intermediates_mean[name]
                    ) / self.n_batch
                    self.forward_intermediates_mean[name] += mean_update

                    second_moment_update = (
                        second_moment - self.forward_intermediates_2nd_moment[name]
                    ) / self.n_batch 
                    self.forward_intermediates_2nd_moment[name] += second_moment_update
                    
                    self.max_forward_update = max(
                        mean_update.abs().max(),
                        second_moment_update.abs().max(),
                        self.max_forward_update
                    )

                    if isinstance(module, Bottleneck):
                        self.activation_probability[name] = (output_detached != 0).float().mean(0)
                    
                module.register_forward_hook(forward_hook)
                
                def backward_hook(module, grad_input, grad_output, name=name):
                    grad_output_detached = grad_output[0].detach()
                    mean = grad_output_detached.mean(0)
                    second_moment = (grad_output_detached ** 2).mean(0)
                    # if isinstance(module, nn.ReLU):
                    #     self.relu_invocation_counter[name] += 1
                    #     name += str(self.relu_invocation_counter[name])

                    mean_update = (
                        mean - self.backward_intermediates_mean[name]
                    ) / self.n_batch
                    self.backward_intermediates_mean[name] += mean_update

                    second_moment_update = (
                        second_moment - self.backward_intermediates_2nd_moment[name]
                    ) / self.n_batch
                    self.backward_intermediates_2nd_moment[name] += second_moment_update

                    self.max_backward_update = max(
                        mean_update.abs().max(),
                        second_moment_update.abs().max(),
                        self.max_backward_update
                    )

                module.register_full_backward_hook(backward_hook)
    
    def _save_gradients(self):
        for name, parameter in self.model.named_parameters():
            mean_update = (
                parameter.grad - self.gradients_mean[name]
            ) / self.n_batch
            self.gradients_mean[name] += mean_update

            second_moment_update = (
                parameter.grad ** 2 - self.gradients_2nd_moment[name]
            ) / self.n_batch
            self.gradients_2nd_moment[name] += second_moment_update

            self.max_grad_update = max(
                mean_update.abs().max(),
                second_moment_update.abs().max(),
                self.max_grad_update
            )