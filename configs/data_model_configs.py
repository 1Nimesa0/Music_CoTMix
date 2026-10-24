def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class MusicGenre(object):
    """
    Configuration for Cross-Environment Music Genre Recognition.
    Source Domain: Clean studio music tracks.
    Target Domain: Environmentally noisy music tracks (acoustic noise, reverb, mic filtering).
    """
    def __init__(self):
        super(MusicGenre, self).__init__()
        # Cross-environment scenarios: clean -> noisy
        self.scenarios = [("clean", "noisy")]
        self.class_names = [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        self.num_classes = 10
        self.sequence_len = 256
        self.input_channels = 1
        
        # Audio Mel-Spectrogram specs
        self.sample_rate = 16000
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.img_size = (128, 256)
        
        # Backbone specs
        self.dino_model_name = 'vit_small_patch16_dinov3'
        self.pretrained = False
        self.feature_dim = 384
        
        # Fallback 1D CNN baseline specs
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.3


class HAR(object):
    def __init__(self):
        super(HAR, self).__init__()
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1


class EEG(object):
    def __init__(self):
        super(EEG, self).__init__()
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.sequence_len = 3000

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2
        self.num_classes = 5

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.scenarios = [("6", "19"), ("7", "18"), ("18", "23"), ("20", "30"), ("35", "31")]
        self.sequence_len = 128

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1


class HHAR(object):
    def __init__(self):
        super(HHAR, self).__init__()
        self.scenarios = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128
        
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
