import os

from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder, ImageNet
from torchvision.models import ResNet50_Weights


class PathMixin:
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path


class ImageNetWithPaths(PathMixin, ImageNet):
    pass


class ImageNetC(PathMixin, ImageFolder):
    def __init__(
            self, root, corruption_type, corruption_severity, 
            transform=ResNet50_Weights.DEFAULT.transforms()):
        root = os.path.join(root, corruption_type, str(corruption_severity))
        super().__init__(root=root, transform=transform)


class ImageNetV2(PathMixin, ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)


def get_imagenet_c_group_loaders(data_dir, batch_size):
    group_loaders = {}
    for corruption in os.listdir(data_dir):
        for severity in range(1, 6):
            group_loaders[f'{corruption}_{severity}'] = DataLoader(
                ImageNetC(data_dir, corruption, severity),
                batch_size=batch_size,
                shuffle=True
            )
            
    return group_loaders 