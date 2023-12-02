import argparse
import random

import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights

from datasets import get_group_loaders 
from stats.stats_manager import StatsManager

MODEL_PATHS = {
	'fmow': 'models/FMoW/resnet50/pretrained/base_model_1-50.pt',
	'camelyon17': 'models/Camelyon17/resnet50/pretrained/base_model_10-5.pt',
	'rxrx1': 'models/RxRx1/resnet50/pretrained/base_model_10-90.pt',
	'iwildcam': 'models/iWildCam/resnet50/pretrained/base_model_1-5.pt'
}


def load_model(dataset_name):
	if dataset_name in ['imagenet-c']:
		return resnet50(weights=ResNet50_Weights)
	elif dataset_name in MODEL_PATHS:
		return torch.load(MODEL_PATHS[dataset_name])['model'].module
	else:
		raise ValueError(f'Invalid dataset name {dataset_name}.')


def adapt_model(model, adaptor=None):
	if adaptor is None:
		model.eval()
		return model
	else:
		raise NotImplementedError


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--out-dir', default='resnet_intermediates')
	parser.add_argument('--batch-size', default=200)
	args = parser.parse_args()

	print('Loading data loaders...')
	group_loaders = get_group_loaders(args.dataset, batch_size=args.batch_size)
	
	for group_id, data_loader in group_loaders.items():
		print(group_id)
		print('Loading model...')
		model = load_model(args.dataset)
		print('Adapting model...')
		model = adapt_model(model)
		stats_manager = StatsManager(model, data_loader, 'cuda')
		stats_manager.collect_stats()
		stats_manager.save_stats(
			f'{args.out_dir.strip()}/{args.dataset}/{group_id}')