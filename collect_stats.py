import argparse
import random

import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights

from datasets import get_group_loaders 
from stats.stats_manager import StatsManager


def load_model(dataset_name):
	return resnet50(weights=ResNet50_Weights)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	args = parser.parse_args()

	print('Loading model...')
	model = load_model(args.dataset)
	print('Loading data loaders...')
	group_loaders = get_group_loaders(args.dataset, batch_size=100)
	
	for group_id, data_loader in group_loaders.items():
		print(group_id)
		stats_manager = StatsManager(model, data_loader, 'cuda')
		stats_manager.collect_stats()
		stats_manager.save_stats(f'{args.dataset}_{group_id}')
		breakpoint()