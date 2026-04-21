import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Dict, List
import pandas as pd


class BaseDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        handle_missing: bool = True
    ):
        self.data_path = data_path
        self.normalize = normalize
        self.handle_missing = handle_missing
        
        self.data = None
        self.labels = None
        self.missing_mask = None
        self.partition_ids = None
        self.timestamps = None
        
        self._load_data()
    
    def _load_data(self):
        raise NotImplementedError
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        features_min = np.nanmin(features, axis=0)
        features_max = np.nanmax(features, axis=0)
        
        range_val = features_max - features_min
        range_val[range_val == 0] = 1.0
        
        normalized = (features - features_min) / range_val
        
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def _handle_missing_values(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        missing_mask = ~data.isna().values
        
        data_filled = data.fillna(data.median())
        
        data_filled = data_filled.fillna(0)
        
        return data_filled.values.astype(np.float32), missing_mask
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = {
            'features': torch.FloatTensor(self.data[idx]),
            'label': torch.FloatTensor([self.labels[idx]]).squeeze().long(),
        }
        
        if self.missing_mask is not None:
            sample['missing_mask'] = torch.BoolTensor(self.missing_mask[idx])
        
        if self.partition_ids is not None:
            sample['partition_ids'] = torch.LongTensor(self.partition_ids[idx])
        
        if self.timestamps is not None:
            sample['timestamps'] = torch.FloatTensor(self.timestamps[idx])
        
        return sample
    
    def get_num_features(self) -> int:
        return self.data.shape[1]
    
    def get_num_partitions(self) -> int:

        if self.partition_ids is not None:
            return int(np.max(self.partition_ids)) + 1
        return 1
