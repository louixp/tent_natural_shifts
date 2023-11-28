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
    ):
        self.model = model.to(device)
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.n_batches = len(data_loader)
        self.device = device

        self._setup_hooks()
        self.gradients_mean = dict()
        self.gradients_2nd_moment = dict()
    
    def collect_stats(self) -> None:
        for x, y_true, _ in tqdm.tqdm(self.data_loader):
            if len(x) < self.batch_size:
                self.n_batches -= 1
                break
            
            self.relu_invocation_counter = defaultdict(int)
            x.requires_grad = True # hack for gradient checkpointing
            y_pred = checkpoint(self.model, x.to(self.device))
            loss = F.cross_entropy(y_pred, y_true.to(self.device))
            loss.backward()

            self._save_gradients()
    
    def save_stats(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        intermediates_attrs = [
            'forward_intermediates_mean', 
            'forward_intermediates_2nd_moment', 
            'backward_intermediates_mean', 
            'backward_intermediates_2nd_moment',
        ]

        for attr in intermediates_attrs:
            with open(os.path.join(path, f'{attr}.pickle'), 'wb') as fp:
                pickle.dump(
                    {
                        k: v / self.batch_size / self.n_batches 
                        for k, v in getattr(self, attr).items()
                    },
                    fp
                )

        gradients_attrs = [
            'gradients_mean',
            'gradients_2nd_moment'
        ]

        for attr in gradients_attrs:
            with open(os.path.join(path, f'{attr}.pickle'), 'wb') as fp:
                pickle.dump(
                    {
                        k: v / self.n_batches 
                        for k, v in getattr(self, attr).items()
                    },
                    fp
                )

    def _setup_hooks(self):
        self.forward_intermediates_mean = dict() 
        self.forward_intermediates_2nd_moment = dict()
        
        self.backward_intermediates_mean = dict()
        self.backward_intermediates_2nd_moment = dict()

        for name, module in self.model.named_modules():
            if isinstance(module, Bottleneck) or name == 'fc':
                def forward_hook(module, input, output, name=name):
                    output_np = output.detach().cpu().numpy()
                    mean = output_np.sum(axis=0)
                    second_moment = np.square(output_np).sum(axis=0)
                    if isinstance(module, nn.ReLU):
                        self.relu_invocation_counter[name] += 1
                        name += str(self.relu_invocation_counter[name])
                    if name not in self.forward_intermediates_mean:
                        self.forward_intermediates_mean[name] = np.zeros_like(mean)
                    if name not in self.forward_intermediates_2nd_moment:
                        self.forward_intermediates_2nd_moment[name] = np.zeros_like(second_moment)

                    self.forward_intermediates_mean[name] += mean
                    self.forward_intermediates_2nd_moment[name] += second_moment
                module.register_forward_hook(forward_hook)
                
                def backward_hook(module, grad_input, grad_output, name=name):
                    grad_output_np = grad_output[0].detach().cpu().numpy()
                    mean = grad_output_np.sum(axis=0)
                    second_moment = np.square(grad_output_np).sum(axis=0)
                    if isinstance(module, nn.ReLU):
                        self.relu_invocation_counter[name] += 1
                        name += str(self.relu_invocation_counter[name])
                    if name not in self.backward_intermediates_mean:
                        self.backward_intermediates_mean[name] = np.zeros_like(mean)
                    if name not in self.backward_intermediates_2nd_moment:
                        self.backward_intermediates_2nd_moment[name] = np.zeros_like(second_moment)

                    self.backward_intermediates_mean[name] += mean
                    self.backward_intermediates_2nd_moment[name] += second_moment
                module.register_full_backward_hook(backward_hook)
    
    def _save_gradients(self):
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad.cpu().numpy()
            if name not in self.gradients_mean:
                self.gradients_mean[name] = np.zeros_like(grad)
            if name not in self.gradients_2nd_moment:
                self.gradients_2nd_moment[name] = np.zeros_like(grad)
            
            self.gradients_mean[name] += grad
            self.gradients_2nd_moment[name] += grad ** 2