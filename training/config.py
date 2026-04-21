import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    input_dim: int = 1
    num_attributes: int = 13
    num_partitions: int = 4
    hidden_dim: int = 512
    measure_dim: int = 16
    use_contrastive: bool = False
    use_robust: bool = False
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'
    scheduler: Optional[str] = 'plateau'
    patience: int = 10
    grad_clip: float = 1.0
    lambda_completion: float = 0.1
    lambda_robust: float = 0.1
    lambda_gp: float = 0.01
    ranking_loss: str = 'bce'
    temperature: float = 1.0
    save_interval: int = 10


@dataclass
class DataConfig:
    dataset_name: str = 'heart_disease'
    data_path: str = './data/raw'
    normalize: bool = True
    handle_missing: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    name: str = 'pstif_wro_experiment'
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = './outputs'
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 1


class Config:

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

    def to_dict(self) -> Dict[str, Any]:

        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment': asdict(self.experiment)
        }

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(
            model=data.get('model'),
            training=data.get('training'),
            data=data.get('data'),
            experiment=data.get('experiment')
        )

    def to_yaml(self, path: str):

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:


    configs = {
        'heart_disease': {
            'model': {'num_partitions': 6},
            'training': {'batch_size': 32, 'learning_rate': 1e-3}
        },
        'sepsis': {
            'model': {'num_partitions': 2},
            'training': {'batch_size': 1024, 'learning_rate': 5e-4}
        },
        'german_credit': {
            'model': {'num_partitions': 7},
            'training': {'batch_size': 64, 'learning_rate': 1e-3}
        },
        'credit_card': {
            'model': {'num_partitions': 5},
            'training': {'batch_size': 2048, 'learning_rate': 1e-3}
        },
        'amazon_electronics': {
            'model': {'num_partitions': 3},
            'training': {'batch_size': 512, 'learning_rate': 1e-3}
        },
        'nyc_taxi': {
            'model': {'num_partitions': 4},
            'training': {'batch_size': 1024, 'learning_rate': 5e-4, 'ranking_loss': 'bce'}
        }
    }

    return configs.get(dataset_name, {})