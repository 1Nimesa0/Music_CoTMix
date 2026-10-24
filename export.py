"""
Export Utility for Cross-Environment Music Model.
Converts PyTorch checkpoint into TorchScript (.pt) and ONNX (.onnx) formats
for high-performance edge and production serving.
"""

import os
import argparse
import torch
from configs.data_model_configs import MusicGenre
from models.models import DinoV3, classifier


class CompleteMusicModel(torch.nn.Module):
    def __init__(self, configs):
        super(CompleteMusicModel, self).__init__()
        self.feature_extractor = DinoV3(configs)
        self.classifier = classifier(configs)

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits


def export_artifacts(checkpoint_path="models/checkpoints/music_cotmix_dinov3.pt",
                     output_dir="models/checkpoints"):
    configs = MusicGenre()
    model = CompleteMusicModel(configs)

    if os.path.exists(checkpoint_path):
        print(f"[Export] Loading weights from '{checkpoint_path}'...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_dict", ckpt)
        fe_dict = {}
        cls_dict = {}
        for k, v in state_dict.items():
            if k.startswith("feature_extractor."):
                fe_dict[k.replace("feature_extractor.", "")] = v
            elif k.startswith("classifier."):
                cls_dict[k.replace("classifier.", "")] = v
            elif k.startswith("0."):
                fe_dict[k.replace("0.", "")] = v
            elif k.startswith("1."):
                cls_dict[k.replace("1.", "")] = v
            else:
                fe_dict[k] = v

        if fe_dict:
            model.feature_extractor.load_state_dict(fe_dict, strict=False)
        if cls_dict:
            model.classifier.load_state_dict(cls_dict, strict=False)
    else:
        print(f"[Export] Warning: '{checkpoint_path}' not found. Exporting base initialized model.")

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    dummy_input = torch.randn(1, 1, 128, 256, dtype=torch.float32)

    # 1. Export TorchScript (No extra dependencies needed)
    ts_path = os.path.join(output_dir, "music_cotmix_dinov3.torchscript.pt")
    try:
        print(f"[Export] Tracing model to TorchScript: '{ts_path}'...")
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        print(f"[Export] TorchScript successfully saved: {ts_path} ({os.path.getsize(ts_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[Export] TorchScript export skipped: {e}")

    # 2. Export ONNX (if onnx package is installed)
    onnx_path = os.path.join(output_dir, "music_cotmix_dinov3.onnx")
    try:
        import onnx
        print(f"[Export] Exporting ONNX graph: '{onnx_path}'...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['logits'],
            dynamic_axes={'mel_spectrogram': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )
        print(f"[Export] ONNX successfully saved: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")
    except ImportError:
        print(f"[Export] Notice: 'onnx' package not installed. Skipping ONNX file generation.")
        print(f"         Run `pip install onnx` if ONNX runtime is required.")
    except Exception as e:
        print(f"[Export] ONNX export failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CoTMix DinoV3 model")
    parser.add_argument("--checkpoint", default="models/checkpoints/music_cotmix_dinov3.pt", type=str)
    parser.add_argument("--output_dir", default="models/checkpoints", type=str)
    args = parser.parse_args()

    export_artifacts(args.checkpoint, args.output_dir)
