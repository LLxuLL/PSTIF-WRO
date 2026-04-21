import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, Tuple, Dict, Any

from .heart_disease import HeartDiseaseDataset
from .sepsis import SepsisDataset
from .german_credit import GermanCreditDataset
from .credit_card import CreditCardDataset
from .amazon_electronics import AmazonElectronicsDataset
from .nyc_taxi import NYCTaxiDataset


class DataLoaderFactory:
    
    DATASET_MAP = {
        'heart_disease': HeartDiseaseDataset,
        'sepsis': SepsisDataset,
        'german_credit': GermanCreditDataset,
        'credit_card': CreditCardDataset,
        'amazon_electronics': AmazonElectronicsDataset,
        'nyc_taxi': NYCTaxiDataset,
    }
    
    @classmethod
    def create_dataset(
        cls,
        dataset_name: str,
        data_path: str,
        **kwargs
    ) -> Any:

        if dataset_name not in cls.DATASET_MAP:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls.DATASET_MAP.keys())}"
            )
        
        dataset_class = cls.DATASET_MAP[dataset_name]
        return dataset_class(data_path, **kwargs)
    
    @classmethod
    def create_data_loaders(
        cls,
        dataset_name: str,
        data_path: str,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 4,
        seed: int = 42,
        **dataset_kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:

        dataset = cls.create_dataset(
            dataset_name,
            data_path,
            **dataset_kwargs
        )
        
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, dataset


def get_data_loader(
    dataset_name: str,
    data_path: str,
    batch_size: int = 32,
    split: str = 'train',
    **kwargs
) -> DataLoader:

    train_loader, val_loader, test_loader, _ = DataLoaderFactory.create_data_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        **kwargs
    )
    
    if split == 'train':
        return train_loader
    elif split == 'val':
        return val_loader
    elif split == 'test':
        return test_loader
    else:
        raise ValueError(f"Unknown split: {split}")


class MultiDatasetLoader:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loaders = {}
        self.datasets = {}
    
    def load_all(self) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:

        for dataset_name, dataset_config in self.config.items():
            print(f"Loading {dataset_name}...")
            
            train_loader, val_loader, test_loader, dataset = \
                DataLoaderFactory.create_data_loaders(
                    dataset_name=dataset_name,
                    **dataset_config
                )
            
            self.loaders[dataset_name] = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            self.datasets[dataset_name] = dataset
            
            print(f"  - Train: {len(train_loader.dataset)} samples")
            print(f"  - Val: {len(val_loader.dataset)} samples")
            print(f"  - Test: {len(test_loader.dataset)} samples")
        
        return self.loaders
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        info = {}
        for name, dataset in self.datasets.items():
            info[name] = {
                'num_samples': len(dataset),
                'num_features': dataset.get_num_features(),
                'num_partitions': dataset.get_num_partitions(),
            }
        return info
