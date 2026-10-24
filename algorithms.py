import torch
import torch.nn as nn
import numpy as np

from models.models import classifier
from models.loss import SupConLoss, ConditionalEntropyLoss, ConfidenceGatedEntropyLoss, NTXentLoss


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    Base class for domain adaptation algorithms.
    Subclasses should implement the update() method.
    """
    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class CoTMix(Algorithm):
    """
    Contrastive Domain Adaptation via Temporal Mixup (CoTMix).
    
    Adapted for Cross-Environment Music Genre Classification with DinoV3 Backbone.
    Core methodology:
    1. Temporal Mixup: Synthesizes intermediate anchor-dominant views across time axis.
    2. Supervised Contrastive Loss (Src): Pulls representations of matching music genres together.
    3. Unsupervised Contrastive Loss (Trg): Mitigates environmental distribution shift by aligning target views.
    4. Gated Conditional Entropy: Enforces decisive class boundaries on target audio without noise collapse.
    """
    def __init__(self, backbone_fe, configs, hparams, device):
        super(CoTMix, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.device = device
        self.hparams = hparams

        # Optimizer: AdamW for Vision Transformer / DinoV3 stabilization
        weight_decay = getattr(hparams, "weight_decay", 1e-2)
        lr = getattr(hparams, "learning_rate", 5e-5)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        batch_size = getattr(hparams, "batch_size", 16)
        self.contrastive_loss = NTXentLoss(device, batch_size=batch_size, temperature=0.2, use_cosine_similarity=True)
        
        # Use Confidence-Gated entropy loss if target threshold specified, else standard entropy
        entropy_threshold = getattr(hparams, "entropy_threshold", 0.60)
        self.entropy_loss = ConfidenceGatedEntropyLoss(threshold=entropy_threshold)
        self.sup_contrastive_loss = SupConLoss(device, temperature=0.2)

    def update(self, src_x, src_y, trg_x):
        """
        Executes one adaptation step on a batch of Source (clean) and Target (noisy) data.
        
        Args:
            src_x: Source audio representations (Mel-spectrogram [B, C, F, T] or 1D series [B, C, L])
            src_y: Source ground-truth genre labels [B]
            trg_x: Target noisy environment audio representations
        """
        mix_ratio = float(getattr(self.hparams, "mix_ratio", 0.80))
        temporal_shift = int(getattr(self.hparams, "temporal_shift", 16))
        h = max(1, temporal_shift // 2)

        # ====== Temporal Mixup =====================
        # For 4D audio mel-spectrogram [B, C, F, T], shift along time axis (dim=-1)
        # to preserve musical pitch and harmonic frequencies.
        src_shifted = torch.mean(torch.stack([torch.roll(trg_x, -i, dims=-1) for i in range(-h, h)], dim=0), dim=0)
        trg_shifted = torch.mean(torch.stack([torch.roll(src_x, -i, dims=-1) for i in range(-h, h)], dim=0), dim=0)

        src_dominant = mix_ratio * src_x + (1.0 - mix_ratio) * src_shifted
        trg_dominant = mix_ratio * trg_x + (1.0 - mix_ratio) * trg_shifted

        # ====== Extract features and calculate logits =====================
        self.optimizer.zero_grad()

        # Source original features & predictions
        src_orig_feat = self.feature_extractor(src_x)
        src_orig_logits = self.classifier(src_orig_feat)

        # Target original features & predictions
        trg_orig_feat = self.feature_extractor(trg_x)
        trg_orig_logits = self.classifier(trg_orig_feat)

        # ====== Primary Losses: Supervised CE (Src) & Gated Entropy (Trg) =====
        src_cls_weight = float(getattr(self.hparams, "src_cls_weight", 1.0))
        trg_entropy_weight = float(getattr(self.hparams, "trg_entropy_weight", 0.05))

        src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
        loss = src_cls_loss * src_cls_weight

        trg_entropy_loss = self.entropy_loss(trg_orig_logits)
        loss += trg_entropy_loss * trg_entropy_weight

        # ====== Auxiliary Contrastive Losses ==================================
        src_supCon_weight = float(getattr(self.hparams, "src_supCon_weight", 0.15))
        trg_cont_weight = float(getattr(self.hparams, "trg_cont_weight", 0.15))

        # Source-dominant mixup
        src_dominant_feat = self.feature_extractor(src_dominant)
        src_dominant_logits = self.classifier(src_dominant_feat)

        # Supervised contrastive learning on Source domain
        src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
        src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
        loss += src_supcon_loss * src_supCon_weight

        # Target-dominant mixup
        trg_dominant_feat = self.feature_extractor(trg_dominant)
        trg_dominant_logits = self.classifier(trg_dominant_feat)

        # Unsupervised contrastive learning on Target domain
        trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
        loss += trg_con_loss * trg_cont_weight

        # ====== Backpropagation ==============================================
        loss.backward()
        self.optimizer.step()

        return {
            'Total_loss': float(loss.item()),
            'src_cls_loss': float(src_cls_loss.item()),
            'trg_entropy_loss': float(trg_entropy_loss.item()),
            'src_supcon_loss': float(src_supcon_loss.item()),
            'trg_con_loss': float(trg_con_loss.item())
        }
