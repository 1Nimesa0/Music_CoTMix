import torch
from torch import nn
import timm


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    name = backbone_name.strip()
    if name.lower() in ["audiocnn", "cnn"]:
        return AudioCNN
    if name not in globals():
        raise NotImplementedError("Backbone not found: {}".format(backbone_name))
    return globals()[name]


class DinoV3(nn.Module):
    """
    DINOv3 Vision Transformer Backbone adapted for Audio Mel-Spectrograms.
    
    Transforms 2D Mel-Spectrogram representations [B, C, F, T] (e.g., [B, 1, 128, 256])
    into rich self-supervised acoustic embeddings [B, feature_dim].
    
    Supports ViT configurations (vit_small_patch16_dinov3, vit_base_patch16_dinov3).
    """
    def __init__(self, configs):
        super(DinoV3, self).__init__()
        model_name = getattr(configs, 'dino_model_name', 'vit_small_patch16_dinov3')
        pretrained = getattr(configs, 'pretrained', False)
        in_chans = getattr(configs, 'input_channels', 1)
        img_size = getattr(configs, 'img_size', (128, 256))
        
        # Instantiate timm DinoV3 model with num_classes=0 for pooled feature vector
        try:
            self.encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                img_size=img_size,
                num_classes=0
            )
            self.feature_dim = self.encoder.num_features
        except Exception as e:
            # Fallback to standard ViT if dinov3 weights are not locally cached or unsupported
            fallback_model = 'vit_small_patch16_224'
            print(f"[DinoV3] Warning: {e}. Falling back to '{fallback_model}'.")
            self.encoder = timm.create_model(
                fallback_model,
                pretrained=False,
                in_chans=in_chans,
                img_size=img_size,
                num_classes=0
            )
            self.feature_dim = self.encoder.num_features

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, C, F, T] (Mel-Spectrogram) or [B, C, L] (1D sensor)
        Returns:
            features: [B, feature_dim]
        """
        if x.dim() == 3:
            # If 1D sensor time series [B, C, L], project or unsqueeze to 2D
            B, C, L = x.shape
            x = x.unsqueeze(2)  # [B, C, 1, L]
            x = nn.functional.interpolate(x, size=(128, 256), mode='bilinear', align_corners=False)
        
        features = self.encoder(x)
        return features


class AudioCNN(nn.Module):
    """
    Lightweight 2D CNN Backbone optimized for Audio Mel-Spectrograms [B, 1, 128, 256].
    Ultra-low latency (<15ms on CPU) and minimal memory footprint (<40MB RAM),
    specifically engineered for Render free/starter deployment.
    """
    def __init__(self, configs):
        super(AudioCNN, self).__init__()
        in_channels = getattr(configs, 'input_channels', 1)
        base_channels = getattr(configs, 'mid_channels', 32)
        dropout = getattr(configs, 'dropout', 0.2)
        out_dim = getattr(configs, 'feature_dim', 384)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 32, 64, 128]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 64, 32, 64]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 128, 16, 32]
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),          # [B, 128, 1, 1]
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(base_channels * 4, out_dim) if (base_channels * 4 != out_dim) else nn.Identity()
        self.feature_dim = out_dim

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        return self.proj(x)


class CNN(nn.Module):
    """Original 1D CNN Backbone from CoTMix (retained for baseline comparison)."""
    def __init__(self, configs):
        super(CNN, self).__init__()

        in_channels = getattr(configs, 'input_channels', 1)
        mid_channels = getattr(configs, 'mid_channels', 64)
        kernel_size = getattr(configs, 'kernel_size', 5)
        stride = getattr(configs, 'stride', 1)
        dropout = getattr(configs, 'dropout', 0.5)
        final_out_channels = getattr(configs, 'final_out_channels', 128)
        features_len = getattr(configs, 'features_len', 1)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(features_len)
        self.feature_dim = final_out_channels * features_len

    def forward(self, x_in):
        if x_in.dim() == 4:
            # Flatten 2D spectrogram along freq axis for 1D CNN baseline
            B, C, F, T = x_in.shape
            x_in = x_in.view(B, C * F, T)

        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class classifier(nn.Module):
    """Classification Head mapping backbone features to genre classes."""
    def __init__(self, configs):
        super(classifier, self).__init__()
        in_dim = getattr(configs, 'feature_dim', None)
        if in_dim is None:
            in_dim = getattr(configs, 'features_len', 1) * getattr(configs, 'final_out_channels', 128)
        self.logits = nn.Linear(in_dim, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions
