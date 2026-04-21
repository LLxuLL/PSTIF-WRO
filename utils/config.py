"""
配置管理模块
"""

import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 10
    num_attributes: int = 10
    num_partitions: int = 5
    hidden_dim: int = 1024
    measure_dim: int = 32
    use_contrastive: bool = True
    use_robust: bool = True
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 200
    batch_size: int = 2048
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'
    scheduler: Optional[str] = 'plateau'
    patience: int = 20
    grad_clip: Optional[float] = 1.0
    
    # 损失权重
    lambda_completion: float = 0.01
    lambda_robust: float = 0.01
    lambda_gp: float = 0.001
    
    # 排序损失
    ranking_loss: str = 'bce'
    temperature: float = 1.0


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = 'nyc_taxi'
    data_path: str = './data'
    normalize: bool = True
    handle_missing: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = 'pstif_wro_experiment'
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = './outputs'
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    
    # 日志
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 1


class Config:
    """
    配置管理类
    """
    
    def __init__(
        self,
        model: Optional[Dict] = None,
        training: Optional[Dict] = None,
        data: Optional[Dict] = None,
        experiment: Optional[Dict] = None
    ):
        self.model = ModelConfig(**(model or {}))
        self.training = TrainingConfig(**(training or {}))
        self.data = DataConfig(**(data or {}))
        self.experiment = ExperimentConfig(**(experiment or {}))
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=config_dict.get('model', {}),
            training=config_dict.get('training', {}),
            data=config_dict.get('data', {}),
            experiment=config_dict.get('experiment', {})
        )
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """从JSON文件加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=config_dict.get('model', {}),
            training=config_dict.get('training', {}),
            data=config_dict.get('data', {}),
            experiment=config_dict.get('experiment', {})
        )
    
    def to_yaml(self, path: str):
        """保存为YAML文件"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment': asdict(self.experiment)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_json(self, path: str):
        """保存为JSON文件"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment': asdict(self.experiment)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment': asdict(self.experiment)
        }
    
    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if '.' in key:
                # 嵌套更新，如 'model.hidden_dim'
                parts = key.split('.')
                obj = getattr(self, parts[0])
                for part in parts[1:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"Config(model={self.model}, training={self.training}, data={self.data}, experiment={self.experiment})"


# 预定义配置
DATASET_CONFIGS = {
    'heart_disease': {
        'model': {
            'input_dim': 13,
            'num_attributes': 13,
            'num_partitions': 6,
            'hidden_dim': 32,
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 5e-4,
            'epochs': 200,
        }
    },
    'sepsis': {
        'model': {
            'input_dim': 3,
            'num_attributes': 3,
            'num_partitions': 2,
            'hidden_dim': 64,
        },
        'training': {
            'batch_size': 256,
            'learning_rate': 1e-3,
            'epochs': 50,
        }
    },
    'german_credit': {
        'model': {
            'input_dim': 20,
            'num_attributes': 20,
            'num_partitions': 7,
            'hidden_dim': 32,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 5e-4,
            'epochs': 300,
        }
    },
    'credit_card': {
        'model': {
            'input_dim': 30,
            'num_attributes': 30,
            'num_partitions': 5,
            'hidden_dim': 128,
        },
        'training': {
            'batch_size': 512,
            'learning_rate': 1e-3,
            'epochs': 20,
        }
    },
    'amazon_electronics': {
        'model': {
            'input_dim': 9,
            'num_attributes': 9,
            'num_partitions': 3,
            'hidden_dim': 128,
        },
        'training': {
            'batch_size': 256,
            'learning_rate': 1e-3,
            'epochs': 10,
        }
    },
    'nyc_taxi': {
        'model': {
            'input_dim': 11,
            'num_attributes': 11,
            'num_partitions': 4,
            'hidden_dim': 256,
        },
        'training': {
            'batch_size': 256,
            'learning_rate': 5e-4,
            'epochs': 50,
        }
    },
}


def get_dataset_config(dataset_name: str) -> Dict:
    """获取数据集预定义配置"""
    return DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['heart_disease'])
