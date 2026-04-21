import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    average_precision_score, ndcg_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from utils.logger import get_logger


class Evaluator:
    """
    PSTIF-WRO Evaluator
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        logger = None
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.logger = logger or get_logger('Evaluator')

        self.results = {}

    def evaluate(self) -> Dict[str, float]:

        self.model.eval()

        all_scores = []
        all_robust_scores = []
        all_labels = []
        all_predictions = []

        self.logger.info("Running evaluation on test set...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)

                batch_size = features.shape[0]
                seq_len = features.shape[1]

                if 'partition_ids' in batch:
                    partition_ids = batch['partition_ids'].to(self.device)
                else:
                    partition_ids = torch.zeros(
                        batch_size, seq_len,
                        dtype=torch.long, device=self.device
                    )

                if 'missing_mask' in batch:
                    missing_mask = batch['missing_mask'].to(self.device)
                else:
                    missing_mask = torch.ones(
                        batch_size, seq_len,
                        dtype=torch.bool, device=self.device
                    )

                timestamps = batch.get('timestamps', None)
                if timestamps is not None:
                    timestamps = timestamps.to(self.device)

                results = self.model(
                    features,
                    partition_ids,
                    timestamps=timestamps,
                    missing_mask=missing_mask
                )

                scores = results['scores']
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                predictions = (scores > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())

                if 'robust_scores' in results:
                    all_robust_scores.extend(results['robust_scores'].cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Processed {batch_idx + 1}/{len(self.test_loader)} batches")

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        metrics = self._compute_metrics(all_labels, all_scores, all_predictions)

        if len(all_robust_scores) > 0:
            all_robust_scores = np.array(all_robust_scores)
            robust_predictions = (all_robust_scores > 0.5).astype(int)
            robust_metrics = self._compute_metrics(
                all_labels,
                all_robust_scores,
                robust_predictions,
                prefix='robust_'
            )
            metrics.update(robust_metrics)

        self.results = {
            'scores': all_scores,
            'labels': all_labels,
            'predictions': all_predictions,
            'metrics': metrics
        }

        self._log_metrics(metrics)

        return metrics

    def _compute_metrics(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        predictions: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:

        metrics = {}

        # AUC
        try:
            metrics[f'{prefix}auc'] = roc_auc_score(labels, scores)
        except Exception as e:
            self.logger.warning(f"Could not compute AUC: {e}")
            metrics[f'{prefix}auc'] = 0.5

        metrics[f'{prefix}accuracy'] = accuracy_score(labels, predictions)

        metrics[f'{prefix}precision'] = precision_score(
            labels, predictions, zero_division=0
        )

        metrics[f'{prefix}recall'] = recall_score(
            labels, predictions, zero_division=0
        )

        metrics[f'{prefix}f1'] = f1_score(labels, predictions, zero_division=0)

        try:
            metrics[f'{prefix}ap'] = average_precision_score(labels, scores)
        except Exception as e:
            self.logger.warning(f"Could not compute AP: {e}")
            metrics[f'{prefix}ap'] = 0.0

        # NDCG
        try:
            metrics[f'{prefix}ndcg@5'] = ndcg_score([labels], [scores], k=5)
            metrics[f'{prefix}ndcg@10'] = ndcg_score([labels], [scores], k=10)
        except Exception as e:
            self.logger.warning(f"Could not compute NDCG: {e}")
            metrics[f'{prefix}ndcg@5'] = 0.0
            metrics[f'{prefix}ndcg@10'] = 0.0

        return metrics

    def _log_metrics(self, metrics: Dict[str, float]):

        self.logger.info("=" * 60)
        self.logger.info("Evaluation Results")
        self.logger.info("=" * 60)

        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"{key:20s}: {value:.4f}")
            else:
                self.logger.info(f"{key:20s}: {value}")

        self.logger.info("=" * 60)

    def plot_confusion_matrix(self, save_path: Optional[str] = None):

        if 'labels' not in self.results or 'predictions' not in self.results:
            self.logger.warning("No results available. Run evaluate() first.")
            return

        try:
            cm = confusion_matrix(
                self.results['labels'],
                self.results['predictions']
            )

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Confusion matrix saved to {save_path}")
            else:
                plt.show()

            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")

    def plot_roc_curve(self, save_path: Optional[str] = None):

        from sklearn.metrics import roc_curve

        if 'labels' not in self.results or 'scores' not in self.results:
            self.logger.warning("No results available. Run evaluate() first.")
            return

        try:
            fpr, tpr, _ = roc_curve(
                self.results['labels'],
                self.results['scores']
            )
            auc = roc_auc_score(
                self.results['labels'],
                self.results['scores']
            )

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"ROC curve saved to {save_path}")
            else:
                plt.show()

            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {e}")

    def adversarial_robustness_test(
        self,
        epsilon: float = 0.1,
        num_steps: int = 10
    ) -> Dict[str, float]:

        self.model.eval()

        original_scores = []
        adversarial_scores = []
        labels = []

        self.logger.info("Starting adversarial robustness test...")

        for batch_idx, batch in enumerate(self.test_loader):
            features = batch['features'].to(self.device)
            batch_labels = batch['label'].to(self.device)

            batch_size = features.shape[0]
            seq_len = features.shape[1]

            if 'partition_ids' in batch:
                partition_ids = batch['partition_ids'].to(self.device)
            else:
                partition_ids = torch.zeros(
                    batch_size, seq_len,
                    dtype=torch.long, device=self.device
                )

            if 'missing_mask' in batch:
                missing_mask = batch['missing_mask'].to(self.device)
            else:
                missing_mask = torch.ones(
                    batch_size, seq_len,
                    dtype=torch.bool, device=self.device
                )

            with torch.no_grad():
                results = self.model(
                    features,
                    partition_ids,
                    missing_mask=missing_mask
                )
                original_scores.extend(results['scores'].cpu().numpy())

            features_adv = self._pgd_attack(
                features,
                batch_labels,
                partition_ids,
                missing_mask,
                epsilon,
                num_steps
            )

            with torch.no_grad():
                results_adv = self.model(
                    features_adv,
                    partition_ids,
                    missing_mask=missing_mask
                )
                adversarial_scores.extend(results_adv['scores'].cpu().numpy())

            labels.extend(batch_labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Adversarial test: {batch_idx + 1}/{len(self.test_loader)} batches")

        original_auc = roc_auc_score(labels, original_scores)
        adversarial_auc = roc_auc_score(labels, adversarial_scores)

        robustness_metrics = {
            'original_auc': original_auc,
            'adversarial_auc': adversarial_auc,
            'auc_drop': original_auc - adversarial_auc,
            'robustness_ratio': adversarial_auc / original_auc if original_auc > 0 else 0
        }

        self.logger.info("Adversarial Robustness Test Results:")
        for key, value in robustness_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

        return robustness_metrics

    def _pgd_attack(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        partition_ids: torch.Tensor,
        missing_mask: torch.Tensor,
        epsilon: float,
        num_steps: int
    ) -> torch.Tensor:

        features_adv = features.clone().detach().requires_grad_(True)
        step_size = epsilon / num_steps

        for _ in range(num_steps):
            if features_adv.grad is not None:
                features_adv.grad.zero_()

            results = self.model(
                features_adv,
                partition_ids,
                missing_mask=missing_mask
            )

            loss = nn.functional.binary_cross_entropy_with_logits(
                results['scores'],
                labels.float()
            )
            loss.backward()

            with torch.no_grad():
                features_adv = features_adv + step_size * features_adv.grad.sign()

                perturbation = torch.clamp(
                    features_adv - features,
                    -epsilon,
                    epsilon
                )
                features_adv = features + perturbation

            features_adv.requires_grad_(True)

        return features_adv.detach()