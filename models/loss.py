import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConditionalEntropyLoss(nn.Module):
    """Vanilla Conditional Entropy Loss for unsupervised target adaptation."""
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class ConfidenceGatedEntropyLoss(nn.Module):
    """
    Confidence-Gated Conditional Entropy Loss for Cross-Environment Adaptation.
    
    Prevents entropy minimization from forcing high-confidence incorrect predictions
    on severely noise-corrupted audio samples (entropy collapse prevention).
    Only samples whose max prediction probability exceeds threshold tau are regularized.
    """
    def __init__(self, threshold=0.60, temperature=1.0):
        super(ConfidenceGatedEntropyLoss, self).__init__()
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        log_probs = F.log_softmax(scaled_logits, dim=1)
        
        # Calculate sample-wise entropy
        sample_entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Gating mask: only retain samples with sufficient prediction confidence
        max_probs, _ = torch.max(probs, dim=1)
        mask = (max_probs >= self.threshold).float()
        
        if mask.sum() > 0:
            return (sample_entropy * mask).sum() / (mask.sum() + 1e-6)
        else:
            # Soft fallback regularization to prevent gradient detachment
            return sample_entropy.mean() * 0.1


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for Contrastive Learning.
    Supports dynamic batch sizes to handle variable batch lengths safely.
    """
    def __init__(self, device, batch_size=32, temperature=0.2, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.default_batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_matrix(self, x, y):
        if self.use_cosine_similarity:
            x_norm = F.normalize(x, p=2, dim=-1)
            y_norm = F.normalize(y, p=2, dim=-1)
            return torch.matmul(x_norm, y_norm.T)
        else:
            return torch.matmul(x, y.T)

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye(2 * batch_size, 2 * batch_size, k=-batch_size)
        l2 = np.eye(2 * batch_size, 2 * batch_size, k=batch_size)
        mask = torch.from_numpy(diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, zis, zjs):
        batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self._get_similarity_matrix(representations, representations)

        # Filter out scores from positive pairs
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon): https://arxiv.org/abs/2004.11362
    Pulls samples of the same genre closer while pushing different genres apart.
    """
    def __init__(self, device, temperature=0.2, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Normalize features for numerical stability
        contrast_feature = F.normalize(contrast_feature, p=2, dim=-1)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_feature = F.normalize(anchor_feature, p=2, dim=-1)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute cosine similarity logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = -self.temperature * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
