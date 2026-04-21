"""
Amazon Electronics (5-core)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class AmazonElectronicsDataset(BaseDataset):
    """
    Amazon Electronics (5-core)
    """
    
    PARTITIONS = {
        'user_features': [0, 1, 2],
        'item_features': [3, 4, 5],
        'interaction': [6, 7, 8]
    }
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        handle_missing: bool = True,
        max_users: int = 10000,
        max_items: int = 5000,
        sequence_length: int = 10
    ):
        self.max_users = max_users
        self.max_items = max_items
        self.sequence_length = sequence_length
        super().__init__(data_path, normalize, handle_missing)
    
    def _load_data(self):

        filepath = f"{self.data_path}/Electronics.csv"
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Creating synthetic data.")
            df = self._create_synthetic_data()

        user_counts = df['user_id'].value_counts()
        item_counts = df['parent_asin'].value_counts()
        
        top_users = user_counts.head(self.max_users).index
        top_items = item_counts.head(self.max_items).index
        
        df = df[
            df['user_id'].isin(top_users) &
            df['parent_asin'].isin(top_items)
        ]

        features_df = self._create_interaction_features(df)

        self.labels = (df['rating'].values >= 4).astype(int)

        if self.handle_missing:
            features, self.missing_mask = self._handle_missing_values(features_df)
        else:
            features = features_df.values
            self.missing_mask = np.ones_like(features, dtype=bool)

        if self.normalize:
            features = self._normalize_features(features)
        
        self.data = features

        self.partition_ids = self._create_partition_ids(features.shape[1])

        timestamps = df['timestamp'].values / 1e9
        self.timestamps = np.tile(timestamps.reshape(-1, 1), (1, features.shape[1]))
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:

        user_ids = pd.Categorical(df['user_id']).codes
        item_ids = pd.Categorical(df['parent_asin']).codes
        

        features = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': df['rating'].values,
            'user_avg_rating': df.groupby('user_id')['rating'].transform('mean').values,
            'item_avg_rating': df.groupby('parent_asin')['rating'].transform('mean').values,
            'user_interaction_count': df.groupby('user_id')['rating'].transform('count').values,
            'item_interaction_count': df.groupby('parent_asin')['rating'].transform('count').values,
            'hour_of_day': pd.to_datetime(df['timestamp'], unit='ms').dt.hour.values,
            'day_of_week': pd.to_datetime(df['timestamp'], unit='ms').dt.dayofweek.values,
        })
        
        return features
    
    def _create_synthetic_data(self) -> pd.DataFrame:

        np.random.seed(42)
        n_samples = 100000
        
        data = {
            'user_id': [f'user_{i}' for i in np.random.randint(0, 10000, n_samples)],
            'parent_asin': [f'item_{i}' for i in np.random.randint(0, 5000, n_samples)],
            'rating': np.random.randint(1, 6, n_samples),
            'timestamp': np.random.randint(1341100800000, 1447286400000, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_partition_ids(self, num_features: int) -> np.ndarray:

        partition_ids = np.zeros(num_features, dtype=int)
        
        for part_id, (part_name, feature_indices) in enumerate(self.PARTITIONS.items()):
            for idx in feature_indices:
                if idx < num_features:
                    partition_ids[idx] = part_id
        
        return np.tile(partition_ids, (len(self.data), 1))
