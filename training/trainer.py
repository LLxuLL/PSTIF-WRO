import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
import time
from tqdm import tqdm

from training.losses import CombinedLoss
from utils.logger import get_logger


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        logger = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger or get_logger('Trainer')
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.criterion = CombinedLoss(
            ranking_loss_type=config.get('ranking_loss', 'bce'),
            temperature=config.get('temperature', 1.0),
            lambda_completion=config.get('lambda_completion', 0.01)
        )
        
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.patience_counter = 0
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'learning_rate': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:

        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:

        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type is None:
            return None
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step', 10),
                gamma=self.config.get('scheduler_gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.get('scheduler_gamma', 0.5),
                patience=self.config.get('scheduler_patience', 5),
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:

        self.model.train()
        
        total_loss = 0.0
        total_ranking_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):

            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            partition_ids = batch.get('partition_ids', None)
            if partition_ids is None:
                partition_ids = torch.zeros(
                    features.shape[0], features.shape[1], 
                    dtype=torch.long, device=self.device
                )
            else:
                partition_ids = partition_ids.to(self.device)
            
            missing_mask = batch.get('missing_mask', None)
            if missing_mask is None:
                missing_mask = torch.ones(
                    features.shape[0], features.shape[1],
                    dtype=torch.bool, device=self.device
                )
            else:
                missing_mask = missing_mask.to(self.device)
            
            timestamps = batch.get('timestamps', None)
            if timestamps is not None:
                timestamps = timestamps.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                results = self.model(
                    features,
                    partition_ids,
                    timestamps=timestamps,
                    missing_mask=missing_mask
                )
                
                losses = self.criterion(results, labels)
                loss = losses['total']
                
                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                
                if self.config.get('grad_clip', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_ranking_loss += losses['ranking'].item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rank': f"{losses['ranking'].item():.4f}"
                })
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            num_batches = 1
        
        avg_losses = {
            'loss': total_loss / num_batches,
            'ranking_loss': total_ranking_loss / num_batches,
        }
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:

        self.model.eval()
        
        total_loss = 0.0
        all_scores = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                partition_ids = batch.get('partition_ids', None)
                if partition_ids is None:
                    partition_ids = torch.zeros(
                        features.shape[0], features.shape[1],
                        dtype=torch.long, device=self.device
                    )
                else:
                    partition_ids = partition_ids.to(self.device)
                
                missing_mask = batch.get('missing_mask', None)
                if missing_mask is None:
                    missing_mask = torch.ones(
                        features.shape[0], features.shape[1],
                        dtype=torch.bool, device=self.device
                    )
                else:
                    missing_mask = missing_mask.to(self.device)
                
                timestamps = batch.get('timestamps', None)
                if timestamps is not None:
                    timestamps = timestamps.to(self.device)
                
                try:
                    results = self.model(
                        features,
                        partition_ids,
                        timestamps=timestamps,
                        missing_mask=missing_mask
                    )
                    
                    losses = self.criterion(results, labels)
                    
                    if not torch.isnan(losses['total']):
                        total_loss += losses['total'].item()
                        num_batches += 1
                    
                    scores = torch.sigmoid(results['scores'])
                    all_scores.extend(scores.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    self.logger.error(f"Error in validation: {e}")
                    continue
        
        from sklearn.metrics import roc_auc_score
        
        if num_batches == 0:
            num_batches = 1
        
        avg_loss = total_loss / num_batches
        
        try:
            if len(set(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_scores)
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        metrics = {
            'val_loss': avg_loss,
            'val_auc': auc,
        }
        
        return metrics
    
    def train(self, epochs: int, save_dir: str = 'checkpoints'):

        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_auc'])
                else:
                    self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_auc'].append(val_metrics['val_auc'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val AUC: {val_metrics['val_auc']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            if val_metrics['val_auc'] > self.best_val_metric:
                self.best_val_metric = val_metrics['val_auc']
                self.patience_counter = 0
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    is_best=True
                )
            else:
                self.patience_counter += 1
            
            patience = self.config.get('patience', 10)
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                )
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, path: str, is_best: bool = False):

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            self.logger.info(f"Best model saved to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.history = checkpoint['history']
        
        self.logger.info(f"Checkpoint loaded from {path}")
