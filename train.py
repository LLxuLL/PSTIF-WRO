import os
import argparse
import torch
import platform

from models.pstif_wro import PSTIFWRO, PSTIFWROConfig
from data.data_loader import DataLoaderFactory
from training.trainer import Trainer
from utils.config import Config, get_dataset_config
from utils.logger import setup_logging
from utils.seed import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train PSTIF-WRO Model')

    parser.add_argument(
        '--dataset',
        type=str,
        default='heart_disease',
        choices=['heart_disease', 'sepsis', 'german_credit',
                 'credit_card', 'amazon_electronics', 'nyc_taxi'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/raw',
        help='Path to dataset'
    )

    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0 if platform.system() == 'Windows' else 4,
        help='Number of data loading workers (0 for Windows)'
    )

    return parser.parse_args()


def main():

    args = parse_args()

    set_random_seed(args.seed)

    if args.config:
        config = Config.from_yaml(args.config)
    else:

        dataset_config = get_dataset_config(args.dataset)
        config = Config(
            model=dataset_config.get('model', {}),
            training=dataset_config.get('training', {})
        )

    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.model.dropout = args.dropout
    config.data.dataset_name = args.dataset
    config.data.data_path = args.data_path
    config.experiment.output_dir = args.output_dir
    config.model.hidden_dim = args.hidden_dim

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

    logger = setup_logging(
        args.output_dir,
        f"pstif_wro_{args.dataset}"
    )

    logger.info("=" * 50)
    logger.info("PSTIF-WRO Training")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Config: {config.to_dict()}")

    logger.info("Loading data...")
    train_loader, val_loader, test_loader, dataset = \
        DataLoaderFactory.create_data_loaders(
            dataset_name=args.dataset,
            data_path=args.data_path,
            batch_size=config.training.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )

    num_features = dataset.get_num_features()
    num_partitions = dataset.get_num_partitions()

    logger.info(f"Number of features: {num_features}")
    logger.info(f"Number of partitions: {num_partitions}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    logger.info("Creating model...")

    model_config = PSTIFWROConfig.get_config({
        'hidden_dim': 512,
        'dropout': 0.2,
        'use_contrastive': False,
        'use_robust': False,
        'gcn_layers': 4,
        'ranking_loss': 'bce',
    })

    model = PSTIFWRO(
        input_dim=1,
        num_attributes=num_features,
        num_partitions=num_partitions,
        hidden_dim=512,
        measure_dim=16,
        use_contrastive=False,
        use_robust=False,
        dropout=0.1,
        config=model_config
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.to_dict()['training'],
        device=args.device,
        logger=logger
    )


    logger.info("Starting training...")
    trainer.train(
        epochs=config.training.epochs,
        save_dir=f"{args.output_dir}/checkpoints"
    )


    config.to_yaml(f"{args.output_dir}/config.yaml")

    logger.info("Training completed!")
    logger.info(f"Best validation AUC: {trainer.best_val_metric:.4f}")


if __name__ == '__main__':
    main()