"""
Credit Card Fraud
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class CreditCardDataset(BaseDataset):
    """
    Credit Card Fraud

    """
    
    PARTITIONS = {
        'time': [0],              # Time
        'pca_components_1': list(range(1, 11)),   # V1-V10
        'pca_components_2': list(range(11, 21)),  # V11-V20
        'pca_components_3': list(range(21, 29)),  # V21-V28
        'amount': [29]            # Amount
    }
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        handle_missing: bool = True,
        balance: bool = False,
        sample_size: Optional[int] = None
    ):
        self.balance = balance
        self.sample_size = sample_size
        super().__init__(data_path, normalize, handle_missing)
    
    def _load_data(self):

        filepath = f"{self.data_path}/creditcard.csv"
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Creating synthetic data.")
            df = self._create_synthetic_data()
        
        self.labels = df['Class'].values
        
        feature_cols = [col for col in df.columns if col != 'Class']
        features_df = df[feature_cols]
        
        if self.balance:
            features_df, self.labels = self._balance_data(features_df, self.labels)
        
        if self.sample_size is not None and len(features_df) > self.sample_size:
            indices = np.random.choice(
                len(features_df),
                self.sample_size,
                replace=False
            )
            features_df = features_df.iloc[indices]
            self.labels = self.labels[indices]
        
        if self.handle_missing:
            features, self.missing_mask = self._handle_missing_values(features_df)
        else:
            features = features_df.values
            self.missing_mask = np.ones_like(features, dtype=bool)
        
        if self.normalize:
            features = self._normalize_features(features)
        
        self.data = features
        
        self.partition_ids = self._create_partition_ids(features.shape[1])
        
        self.timestamps = np.tile(
            np.arange(features.shape[1]),
            (len(self.data), 1)
        ).astype(float)
    
    def _balance_data(
        self,
        features: pd.DataFrame,
        labels: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        fraud_indices = np.where(labels == 1)[0]
        normal_indices = np.where(labels == 0)[0]
        
        n_fraud = len(fraud_indices)
        normal_sample_indices = np.random.choice(
            normal_indices,
            n_fraud * 5,
            replace=False
        )
        
        balanced_indices = np.concatenate([fraud_indices, normal_sample_indices])
        np.random.shuffle(balanced_indices)
        
        return features.iloc[balanced_indices], labels[balanced_indices]
    
    def _create_synthetic_data(self) -> pd.DataFrame:

        np.random.seed(42)
        n_samples = 284807
        
        data = {'Time': np.random.randint(0, 172792, n_samples)}
        
        for i in range(1, 29):
            data[f'V{i}'] = np.random.randn(n_samples)
        
        data['Amount'] = np.random.exponential(88, n_samples)
        data['Class'] = np.random.choice(
            [0, 1],
            n_samples,
            p=[0.99827, 0.00173]
        )
        
        return pd.DataFrame(data)
    
    def _create_partition_ids(self, num_features: int) -> np.ndarray:

        partition_ids = np.zeros(num_features, dtype=int)
        
        for part_id, (part_name, feature_indices) in enumerate(self.PARTITIONS.items()):
            for idx in feature_indices:
                if idx < num_features:
                    partition_ids[idx] = part_id
        
        return np.tile(partition_ids, (len(self.data), 1))
