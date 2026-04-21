"""
German Credit (UCI)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class GermanCreditDataset(BaseDataset):
    """
    German Credit (UCI) :

    1. Status of existing checking account
    2. Duration in month
    3. Credit history
    4. Purpose
    5. Credit amount
    6. Savings account/bonds
    7. Present employment since
    8. Installment rate in percentage of disposable income
    9. Personal status and sex
    10. Other debtors / guarantors
    11. Present residence since
    12. Property
    13. Age in years
    14. Other installment plans
    15. Housing
    16. Number of existing credits at this bank
    17. Job
    18. Number of people being liable to provide maintenance for
    19. Telephone
    20. Foreign worker
    """
    
    PARTITIONS = {
        'financial_status': [0, 5],      # 支票账户状态, 储蓄账户
        'credit_history': [2, 3],         # 信用历史, 贷款目的
        'loan_details': [1, 4, 7],        # 期限, 金额, 分期付款率
        'employment': [6, 10, 16],        # 就业状况, 居住时间, 工作
        'demographic': [8, 12, 17],       # 个人状态, 年龄, 抚养人数
        'assets': [11, 14],               # 财产, 住房
        'other': [9, 13, 15, 18, 19]      # 其他债务人, 其他分期计划等
    }
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        handle_missing: bool = True,
        encode_categorical: bool = True
    ):
        self.encode_categorical = encode_categorical
        super().__init__(data_path, normalize, handle_missing)
    
    def _load_data(self):
        filepath = f"{self.data_path}/german.data"
        
        try:
            df = pd.read_csv(filepath, sep=' ', header=None)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Creating synthetic data.")
            df = self._create_synthetic_data()
        
        self.labels = (df.iloc[:, -1].values == 1).astype(int)
        
        features_df = df.iloc[:, :-1]
        
        if self.encode_categorical:
            features_df = self._encode_features(features_df)
        
        if self.handle_missing:
            features, self.missing_mask = self._handle_missing_values(features_df)
        else:
            features = features_df.values
            self.missing_mask = np.ones_like(features, dtype=bool)
        
        if self.normalize:
            features = self._normalize_features(features)
        
        self.data = features
        
        self.partition_ids = self._create_partition_ids(features.shape[1])
        
        self.timestamps = None
    
    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        # 对分类变量进行标签编码
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """创建合成数据"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {}
        for i in range(20):
            if i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]:
                # 分类特征
                data[i] = np.random.randint(0, 5, n_samples)
            else:
                # 数值特征
                data[i] = np.random.randint(1, 100, n_samples)
        
        # 标签
        data[20] = np.random.randint(1, 3, n_samples)
        
        return pd.DataFrame(data)
    
    def _create_partition_ids(self, num_features: int) -> np.ndarray:
        """创建分区ID"""
        partition_ids = np.zeros(num_features, dtype=int)
        
        for part_id, (part_name, feature_indices) in enumerate(self.PARTITIONS.items()):
            for idx in feature_indices:
                if idx < num_features:
                    partition_ids[idx] = part_id
        
        return np.tile(partition_ids, (len(self.data), 1))
