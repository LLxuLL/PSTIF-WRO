import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import os


def plot_measure_distribution(
    measures: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):

    if isinstance(measures, torch.Tensor):
        measures = measures.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    component_names = ['Membership (μ)', 'Non-membership (ν)', 'Hesitation (π)']
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        if labels is not None:
            for label in np.unique(labels):
                mask = labels == label
                ax.hist(measures[mask, i], bins=30, alpha=0.5, label=f'Class {label}')
            ax.legend()
        else:
            ax.hist(measures[:, i], bins=30, alpha=0.7)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {name}')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    if 'val_auc' in history:
        axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'], label='Learning Rate', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
    
    if 'val_accuracy' in history:
        axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_simplex_projection(
    measures: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):

    if isinstance(measures, torch.Tensor):
        measures = measures.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    sqrt3_2 = np.sqrt(3) / 2
    
    x = measures[:, 1] + measures[:, 2] / 2
    y = measures[:, 2] * sqrt3_2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(x, y, c=labels, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Class')
    else:
        ax.scatter(x, y, alpha=0.6, s=20)
    
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, sqrt3_2, 0]
    ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
    
    ax.text(-0.05, -0.05, 'μ', fontsize=14, fontweight='bold')
    ax.text(1.05, -0.05, 'ν', fontsize=14, fontweight='bold')
    ax.text(0.45, sqrt3_2 + 0.05, 'π', fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title('IF-Measure Distribution on Simplex Δ²')
    ax.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_wasserstein_distance_matrix(
    measures: torch.Tensor,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):

    if isinstance(measures, torch.Tensor):
        measures = measures.cpu().numpy()
    
    num_measures = len(measures)
    distance_matrix = np.zeros((num_measures, num_measures))
    
    for i in range(num_measures):
        for j in range(num_measures):
            distance_matrix[i, j] = np.linalg.norm(measures[i] - measures[j])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=labels or range(num_measures),
        yticklabels=labels or range(num_measures),
        ax=ax
    )
    
    ax.set_title('Wasserstein Distance Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
