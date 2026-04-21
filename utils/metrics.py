import torch
import numpy as np
from typing import List, Union
from sklearn.metrics import roc_auc_score


def compute_auc(
    scores: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray]
) -> float:

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5


def compute_ndcg(
    scores: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    k: int = 10
) -> float:

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    order = np.argsort(scores)[::-1][:k]
    sorted_labels = labels[order]
    
    dcg = 0.0
    for i, label in enumerate(sorted_labels):
        dcg += (2 ** label - 1) / np.log2(i + 2)
    
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = 0.0
    for i, label in enumerate(ideal_labels):
        idcg += (2 ** label - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = 'listmle'
) -> torch.Tensor:

    if loss_type == 'listmle':

        sorted_indices = torch.argsort(labels, descending=True)
        sorted_scores = scores[sorted_indices]
        
        loss = torch.tensor(0.0, device=scores.device)
        for i in range(len(sorted_scores)):
            loss += torch.logsumexp(sorted_scores[i:], dim=0) - sorted_scores[i]
        
        return loss / len(scores)
    
    elif loss_type == 'listnet':

        pred_probs = torch.softmax(scores, dim=0)
        true_probs = torch.softmax(labels.float(), dim=0)
        
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-8))
        
        return loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_precision_at_k(
    scores: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    k: int = 10
) -> float:

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_labels = labels[top_k_indices]
    
    return np.sum(top_k_labels) / k


def compute_recall_at_k(
    scores: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    k: int = 10
) -> float:

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_labels = labels[top_k_indices]
    
    total_positives = np.sum(labels)
    if total_positives == 0:
        return 0.0
    
    return np.sum(top_k_labels) / total_positives
