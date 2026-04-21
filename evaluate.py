import os
import sys
import argparse
import torch
import platform
import json
import traceback

# 强制打印启动信息
print("=" * 60)
print("DEBUG: Script started")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

try:
    from models.pstif_wro import PSTIFWRO, PSTIFWROConfig
    from data.data_loader import DataLoaderFactory
    from training.evaluator import Evaluator
    from utils.logger import setup_logging
    from utils.seed import set_random_seed
    print("DEBUG: All imports successful")
except Exception as e:
    print(f"DEBUG: Import error: {e}")
    traceback.print_exc()
    sys.exit(1)


def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate PSTIF-WRO Model')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
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
        '--batch_size',
        type=int,
        default=512,
        help='Batch size'
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
        default='./outputs/results',
        help='Output directory'
    )
    parser.add_argument(
        '--adversarial_test',
        action='store_true',
        help='Run adversarial robustness test'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )

    # 模型架构参数
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--measure_dim',
        type=int,
        default=16,
        help='Measure dimension'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )

    args = parser.parse_args()
    print(f"DEBUG: Args parsed: {args}")
    return args


def main():

    print("DEBUG: Entering main()")

    try:
        args = parse_args()

        print(f"Checkpoint: {args.checkpoint}")
        print(f"Dataset: {args.dataset}")
        print(f"Hidden dim: {args.hidden_dim}")
        print(f"Measure dim: {args.measure_dim}")
        print(f"Device: {args.device}")

        print("DEBUG: Setting random seed...")
        set_random_seed(42)


        print(f"DEBUG: Creating output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

        print("DEBUG: Setting up logging...")
        logger = setup_logging(
            args.output_dir,
            f"evaluate_{args.dataset}"
        )

        logger.info("=" * 60)
        logger.info("PSTIF-WRO Evaluation")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Hidden dim: {args.hidden_dim}")
        logger.info(f"Measure dim: {args.measure_dim}")

        logger.info(f"Loading checkpoint from {args.checkpoint}")
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)

        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        logger.info(f"Checkpoint loaded, keys: {checkpoint.keys()}")

        logger.info("Loading data...")
        try:
            _, _, test_loader, dataset = DataLoaderFactory.create_data_loaders(
                dataset_name=args.dataset,
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            logger.info(f"Data loaded, test samples: {len(test_loader.dataset)}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            traceback.print_exc()
            sys.exit(1)

        num_features = dataset.get_num_features()
        num_partitions = dataset.get_num_partitions()

        logger.info(f"Number of features: {num_features}")
        logger.info(f"Number of partitions: {num_partitions}")

        logger.info(f"Creating model with hidden_dim={args.hidden_dim}, measure_dim={args.measure_dim}")
        try:
            model_config = PSTIFWROConfig.get_config({
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'use_contrastive': False,
                'use_robust': False,
                'gcn_layers': 3,
                'use_temporal': True,
            })

            model = PSTIFWRO(
                input_dim=1,
                num_attributes=num_features,
                num_partitions=num_partitions,
                hidden_dim=args.hidden_dim,
                measure_dim=args.measure_dim,
                use_contrastive=False,
                use_robust=False,
                dropout=args.dropout,
                config=model_config
            )
            logger.info("Model created successfully")
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            traceback.print_exc()
            sys.exit(1)

        logger.info("Loading model weights...")
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            logger.error(f"Checkpoint keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")
            traceback.print_exc()
            sys.exit(1)

        model.to(args.device)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        logger.info("Starting evaluation...")
        try:
            evaluator = Evaluator(
                model=model,
                test_loader=test_loader,
                device=args.device,
                logger=logger
            )

            metrics = evaluator.evaluate()
            logger.info("Evaluation completed successfully")

            results = {
                'dataset': args.dataset,
                'checkpoint': args.checkpoint,
                'metrics': {k: float(v) if isinstance(v, (int, float)) else str(v)
                           for k, v in metrics.items()}
            }

            results_file = os.path.join(args.output_dir, f'{args.dataset}_test_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

            cm_path = os.path.join(args.output_dir, f'{args.dataset}_confusion_matrix.png')
            evaluator.plot_confusion_matrix(save_path=cm_path)

            roc_path = os.path.join(args.output_dir, f'{args.dataset}_roc_curve.png')
            evaluator.plot_roc_curve(save_path=roc_path)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            traceback.print_exc()
            sys.exit(1)

        logger.info("Evaluation script completed successfully")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("DEBUG: __main__ block entered")
    main()
    print("DEBUG: main() returned")