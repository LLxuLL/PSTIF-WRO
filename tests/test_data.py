import torch
import pytest
from data.data_loader import DataLoaderFactory


def test_dataset_creation():

    
    try:
        dataset = DataLoaderFactory.create_dataset(
            'heart_disease',
            './data/raw',
            dataset_name='cleveland'
        )
        
        assert len(dataset) > 0
        assert dataset.get_num_features() > 0
        assert dataset.get_num_partitions() > 0
        
        sample = dataset[0]
        assert 'features' in sample
        assert 'label' in sample
        
    except FileNotFoundError:
        pytest.skip("Data file not found, skipping test")


def test_data_loader():
    try:
        train_loader, val_loader, test_loader, dataset = \
            DataLoaderFactory.create_data_loaders(
                'heart_disease',
                './data/raw',
                batch_size=4
            )
        
        assert len(train_loader) > 0
        
        batch = next(iter(train_loader))
        assert 'features' in batch
        assert 'label' in batch
        
    except FileNotFoundError:
        pytest.skip("Data file not found, skipping test")


if __name__ == '__main__':
    test_dataset_creation()
    test_data_loader()
    print("All data tests passed!")
