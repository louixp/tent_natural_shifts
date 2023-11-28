from collections import defaultdict

import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_train_loader 
from wilds.datasets.wilds_dataset import WILDSSubset

# TODO: Are these good splits?
GROUPBY_FIELDS = {
	'fmow': ['region', 'year'],
	'camelyon17': ['hospital', 'slide'],
	'rxrx1': ['cell_type', 'site'],
	'iwildcam': ['location', 'year'],
}

def get_group_loaders(dataset_name, groupby_fields, batch_size):
	dataset = get_dataset(dataset=dataset_name)
	grouper = CombinatorialGrouper(dataset, groupby_fields)

	test_data = dataset.get_subset(
		'test',
		transform=transforms.Compose(
			[
				transforms.Resize((224, 224)), 
				transforms.ToTensor(),
				transforms.Normalize(
					[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]
		),
	)
	
	group_id_to_sample_ids = defaultdict(list)
	for i, (_, _, metadata) in enumerate(test_data):
		group_id = grouper.metadata_to_group(metadata.unsqueeze(0)).item()
		group_id_to_sample_ids[group_id].append(i)
	
	return {
		grouper.group_str(group_id): get_train_loader(
			'standard',
			WILDSSubset(test_data, sample_ids, None),
			batch_size=batch_size
		)
		for group_id, sample_ids in group_id_to_sample_ids.items()
	}