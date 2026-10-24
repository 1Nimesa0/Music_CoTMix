def get_hparams_class(dataset_name):
    """Return the hyperparameters class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class MusicGenre(object):
    """
    Hyperparameters optimized for Cross-Environment Music Genre Adaptation with DinoV3 Backbone.
    
    Tuning rationale:
    - learning_rate (5e-5): Smaller than CNN (1e-3) to preserve DinoV3 pre-trained visual/acoustic features.
    - weight_decay (1e-2): Standard weight decay for Vision Transformers.
    - mix_ratio (0.80): High anchor dominance preserves genre-critical musical features.
    - temporal_shift (16): Spectrogram time-axis shift (16 frames ~ 0.5s) to mix temporal acoustic contexts.
    - src_cls_weight (1.0): Anchor supervision on clean source domain.
    - src_supCon_weight (0.15): Encourages intra-genre cohesion and inter-genre separation.
    - trg_cont_weight (0.15): Contrastive alignment on target noisy environment views.
    - trg_entropy_weight (0.05): Regularizes target prediction sharpness under confidence gating.
    - entropy_threshold (0.60): Prevents overconfident predictions on heavily noise-corrupted audio.
    """
    def __init__(self):
        super(MusicGenre, self).__init__()
        self.train_params = {
            'num_epochs': 30,
            'batch_size': 16,
            'weight_decay': 1e-2,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 5e-5,
                'weight_decay': 1e-2,
                'mix_ratio': 0.80,
                'temporal_shift': 16,
                'src_cls_weight': 1.0,
                'src_supCon_weight': 0.15,
                'trg_cont_weight': 0.15,
                'trg_entropy_weight': 0.05,
                'entropy_threshold': 0.60,
                'batch_size': 16,
            }
        }


class HAR(object):
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.9, 
                'temporal_shift': 14, 
                'src_cls_weight': 0.78,
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1, 
                'trg_entropy_weight': 0.05
            }
        }


class EEG(object):
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.79,
                'temporal_shift': 300, 
                'src_cls_weight': 0.96, 
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1, 
                'trg_entropy_weight': 0.05
            }
        }


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.72,
                'temporal_shift': 14, 
                'src_cls_weight': 0.98,
                'src_supCon_weight': 0.1, 
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05,
            }
        }


class HHAR(object):
    def __init__(self):
        super(HHAR, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.52,
                'temporal_shift': 14,
                'src_cls_weight': 0.8,
                'src_supCon_weight': 0.1, 
                'trg_cont_weight': 0.1, 
                'trg_entropy_weight': 0.05, 
            }
        }
