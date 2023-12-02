import os

from . import wilds_datasets
from . import imagenet_datasets

DATA_DIR = 'data/'

def get_group_loaders(name, batch_size):
	if name in wilds_datasets.GROUPBY_FIELDS:
		return wilds_datasets.get_group_loaders(
			name, wilds_datasets.GROUPBY_FIELDS[name], batch_size)
	elif name == 'imagenet-c':
		return imagenet_datasets.get_imagenet_c_group_loaders(
			os.path.join(DATA_DIR, 'imagenet-c'), batch_size)
	else:
		raise ValueError(f'Dataset {name} does not exist.')