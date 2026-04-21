"""
Heart Disease (UCI)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class HeartDiseaseDataset(BaseDataset):
    """
    Heart Disease (UCI)
    - cleveland.data
    - long-beach-va.data
    - switzerland.data
    - hungarian.data
    """
    
    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal'
    ]
    
    PARTITIONS = {
        'demographic': [0, 1],  # age, sex
        'symptoms': [2, 8],      # cp, exang
        'vital_signs': [3, 7],   # trestbps, thalach
        'lab_tests': [4, 5, 12], # chol, fbs, thal
        'ecg': [6, 9, 10],       # restecg, oldpeak, slope
        'angiography': [11]      # ca
    }
    
    def __init__(
        self,
        data_path: str,
        dataset_name: str = 'cleveland',
        normalize: bool = True,
        handle_missing: bool = True,
        binary_classification: bool = True
    ):
        self.dataset_name = dataset_name
        self.binary_classification = binary_classification
        super().__init__(data_path, normalize, handle_missing)
    
    def _load_data(self):
        file_map = {
            'cleveland': 'cleveland.data',
            'long-beach-va': 'long-beach-va.data',
            'switzerland': 'switzerland.data',
            'hungarian': 'hungarian.data'
        }
        
        filename = file_map.get(self.dataset_name, 'cleveland.data')
        filepath = f"{self.data_path}/{filename}"
        
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=self.FEATURE_NAMES + ['target'],
                na_values='?'
            )
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Creating synthetic data.")
            df = self._create_synthetic_data()
        
        self.labels = df['target'].values
        
        if self.binary_classification:
            self.labels = (self.labels > 0).astype(int)
        
        features_df = df.drop('target', axis=1)
        
        if self.handle_missing:
            features, self.missing_mask = self._handle_missing_values(features_df)
        else:
            features = features_df.values
            self.missing_mask = np.ones_like(features, dtype=bool)
        
        if self.normalize:
            features = self._normalize_features(features)
        
        self.data = features
        
        self.partition_ids = self._create_partition_ids()
        
        self.timestamps = None
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n_samples = 303
        
        data = {
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(94, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 3, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_partition_ids(self) -> np.ndarray:
        partition_ids = np.zeros(len(self.FEATURE_NAMES), dtype=int)
        
        for part_id, (part_name, feature_indices) in enumerate(self.PARTITIONS.items()):
            for idx in feature_indices:
                if idx < len(self.FEATURE_NAMES):
                    partition_ids[idx] = part_id
        
        return np.tile(partition_ids, (len(self.data), 1))
