"""
Sepsis Survival (UCI)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class SepsisDataset(BaseDataset):
    """
    Sepsis Survival (UCI)
    - s41598-020-73558-3_sepsis_survival_primary_cohort.csv
    - s41598-020-73558-3_sepsis_survival_study_cohort.csv
    - s41598-020-73558-3_sepsis_survival_validation_cohort.csv
    """
    
    FEATURE_NAMES = [
        'age_years',
        'sex_0male_1female',
        'episode_number'
    ]
    
    PARTITIONS = {
        'demographic': [0, 1],  # age, sex
        'clinical': [2]         # episode_number
    }
    
    def __init__(
        self,
        data_path: str,
        cohort: str = 'primary',
        normalize: bool = True,
        handle_missing: bool = True
    ):
        self.cohort = cohort
        super().__init__(data_path, normalize, handle_missing)
    
    def _load_data(self):
        file_map = {
            'primary': 's41598-020-73558-3_sepsis_survival_primary_cohort.csv',
            'study': 's41598-020-73558-3_sepsis_survival_study_cohort.csv',
            'validation': 's41598-020-73558-3_sepsis_survival_validation_cohort.csv'
        }
        
        filename = file_map.get(self.cohort, file_map['primary'])
        filepath = f"{self.data_path}/{filename}"
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Creating synthetic data.")
            df = self._create_synthetic_data()
        
        self.labels = df['hospital_outcome_1alive_0dead'].values
        
        feature_cols = [col for col in df.columns if col != 'hospital_outcome_1alive_0dead']
        features_df = df[feature_cols]
        
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
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n_samples = 110000
        
        data = {
            'age_years': np.random.randint(18, 100, n_samples),
            'sex_0male_1female': np.random.randint(0, 2, n_samples),
            'episode_number': np.random.randint(1, 10, n_samples),
            'hospital_outcome_1alive_0dead': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_partition_ids(self, num_features: int) -> np.ndarray:
        partition_ids = np.zeros(num_features, dtype=int)
        
        for part_id, (part_name, feature_indices) in enumerate(self.PARTITIONS.items()):
            for idx in feature_indices:
                if idx < num_features:
                    partition_ids[idx] = part_id
        
        return np.tile(partition_ids, (len(self.data), 1))
