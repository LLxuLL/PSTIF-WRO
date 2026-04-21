"""
NYC Taxi Yellow
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class NYCTaxiDataset(BaseDataset):
    """
    NYC Taxi Yellow
    """
    
    PARTITIONS = {
        'temporal': [0, 1, 2, 3],
        'pickup_location': [4, 5],
        'dropoff_location': [6, 7],
        'trip_info': [8, 9]
    }
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        handle_missing: bool = True,
        sample_size: Optional[int] = 100000,
        predict_duration: bool = True
    ):
        self.sample_size = sample_size
        self.predict_duration = predict_duration
        super().__init__(data_path, normalize, handle_missing)
    
    def _load_data(self):
        filepath = f"{self.data_path}/yellowtaxi_data.csv"
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Creating synthetic data.")
            df = self._create_synthetic_data()
        
        if self.sample_size is not None and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            df = df.reset_index(drop=True)
        
        features_df = self._create_spatiotemporal_features(df)
        
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        if self.predict_duration:
            self.labels = df['trip_duration'].values.astype(np.float32)
            median_duration = np.median(self.labels)
            self.labels = (self.labels > median_duration).astype(int)
        else:
            self.labels = features_df['is_rush_hour'].values.astype(int)
        
        if self.handle_missing:
            features, self.missing_mask = self._handle_missing_values(features_df)
        else:
            features = features_df.values.astype(np.float32)
            self.missing_mask = np.ones_like(features, dtype=bool)
        
        if self.normalize:
            features = self._normalize_features(features)
        
        self.data = features
        
        self.partition_ids = self._create_partition_ids(features.shape[1])
        
        self.timestamps = np.tile(
            features_df['hour'].values.reshape(-1, 1),
            (1, features.shape[1])
        ).astype(np.float32)
    
    def _create_spatiotemporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pickup_time = pd.to_datetime(df['pickup_datetime'])
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        
        distance = haversine(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values,
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values
        )
        
        features = pd.DataFrame({
            'hour': pickup_time.dt.hour.values.astype(np.float32),
            'day_of_week': pickup_time.dt.dayofweek.values.astype(np.float32),
            'month': pickup_time.dt.month.values.astype(np.float32),
            'is_weekend': (pickup_time.dt.dayofweek >= 5).astype(np.float32),
            'is_rush_hour': (
                ((pickup_time.dt.hour >= 7) & (pickup_time.dt.hour <= 9)) |
                ((pickup_time.dt.hour >= 17) & (pickup_time.dt.hour <= 19))
            ).astype(np.float32),
            'pickup_latitude': df['pickup_latitude'].values.astype(np.float32),
            'pickup_longitude': df['pickup_longitude'].values.astype(np.float32),
            'dropoff_latitude': df['dropoff_latitude'].values.astype(np.float32),
            'dropoff_longitude': df['dropoff_longitude'].values.astype(np.float32),
            'passenger_count': df['passenger_count'].values.astype(np.float32),
            'distance': distance.astype(np.float32),
            'vendor_id': df['vendor_id'].values.astype(np.float32),
        })
        
        return features
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n_samples = 100000
        
        start_date = pd.Timestamp('2016-01-01')
        end_date = pd.Timestamp('2016-06-30')
        
        random_times = start_date + pd.to_timedelta(
            np.random.randint(0, int((end_date - start_date).total_seconds()), n_samples),
            unit='s'
        )
        
        data = {
            'id': [f'id_{i}' for i in range(n_samples)],
            'vendor_id': np.random.randint(1, 3, n_samples),
            'pickup_datetime': random_times,
            'dropoff_datetime': random_times + pd.to_timedelta(
                np.random.randint(60, 3600, n_samples),
                unit='s'
            ),
            'passenger_count': np.random.randint(1, 7, n_samples),
            'pickup_longitude': np.random.uniform(-74.2, -73.7, n_samples),
            'pickup_latitude': np.random.uniform(40.6, 40.9, n_samples),
            'dropoff_longitude': np.random.uniform(-74.2, -73.7, n_samples),
            'dropoff_latitude': np.random.uniform(40.6, 40.9, n_samples),
            'store_and_fwd_flag': np.random.choice(['Y', 'N'], n_samples),
            'trip_duration': np.random.randint(60, 3600, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_partition_ids(self, num_features: int) -> np.ndarray:
        partition_ids = np.zeros(num_features, dtype=int)
        
        for part_id, (part_name, feature_indices) in enumerate(self.PARTITIONS.items()):
            for idx in feature_indices:
                if idx < num_features:
                    partition_ids[idx] = part_id
        
        return np.tile(partition_ids, (len(self.data), 1))
