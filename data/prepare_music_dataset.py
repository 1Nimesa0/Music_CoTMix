"""
Dataset Preparation Pipeline for Cross-Environment Music Genre Recognition.

Steps:
1. Loads clean WAV files from data/audio/<genre>/.
2. Segments 30-second tracks into 8.192-second clips (131,072 samples @ 16 kHz).
3. Constructs two cross-domain partitions:
   - Source Domain (Clean): Original studio audio -> Log-Mel Spectrogram [1, 128, 256]
   - Target Domain (Noisy): Acoustic environmental distortion -> Log-Mel Spectrogram [1, 128, 256]
4. Splits into Train (80%) and Test (20%) sets.
5. Serializes to PyTorch format (.pt) expected by CoTMix.
"""

import os
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from data.augment_environment import simulate_environmental_audio


GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def extract_log_mel(audio_clip, sr=16000, n_mels=128, n_fft=2048, hop_length=512, target_frames=256):
    """Computes normalized Log-Mel Spectrogram [1, n_mels, target_frames]."""
    mel = librosa.feature.melspectrogram(
        y=audio_clip,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or slice to exact target_frames
    if log_mel.shape[1] < target_frames:
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :target_frames]

    # Standardize to [-1, 1] range for neural networks
    log_mel = (log_mel + 40.0) / 40.0
    log_mel = np.clip(log_mel, -1.0, 1.0)

    # Shape: [1, n_mels, target_frames]
    return np.expand_dims(log_mel.astype(np.float32), axis=0)


def prepare_dataset(raw_audio_dir="data/audio", output_dir="data/MusicGenre", target_sr=16000, max_files_per_genre=None):
    clip_len = 131072  # 8.192 seconds at 16 kHz gives 256 frames with hop_length=512
    os.makedirs(output_dir, exist_ok=True)

    src_samples = []
    src_labels = []
    trg_samples = []
    trg_labels = []

    print(f"[Dataset Preparation] Scanning audio from: {raw_audio_dir}")
    print(f"[Dataset Preparation] Target output folder: {output_dir}")

    total_clips = 0

    for genre in GENRES:
        genre_dir = os.path.join(raw_audio_dir, genre)
        if not os.path.exists(genre_dir):
            print(f"[Warning] Genre folder '{genre_dir}' not found. Skipping.")
            continue

        label = GENRE_TO_IDX[genre]
        files = [f for f in os.listdir(genre_dir) if f.lower().endswith(('.wav', '.mp3'))]
        if max_files_per_genre:
            files = files[:max_files_per_genre]

        print(f" -> Processing '{genre}' ({len(files)} files)...")

        for f in files:
            path = os.path.join(genre_dir, f)
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)  # Convert stereo to mono

                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                # Segment into 8.192s chunks
                num_clips = len(audio) // clip_len
                if num_clips == 0 and len(audio) > 16000:
                    # Pad short audio
                    padded = np.pad(audio, (0, clip_len - len(audio)), mode='wrap')
                    clips = [padded]
                else:
                    clips = [audio[i * clip_len:(i + 1) * clip_len] for i in range(min(num_clips, 3))]

                for clean_clip in clips:
                    # Source Domain: Clean Log-Mel Spectrogram
                    clean_spec = extract_log_mel(clean_clip, sr=target_sr)

                    # Target Domain: Environmentally Perturbed Log-Mel Spectrogram
                    noisy_clip = simulate_environmental_audio(clean_clip, sr=target_sr, profile="random")
                    noisy_spec = extract_log_mel(noisy_clip, sr=target_sr)

                    src_samples.append(clean_spec)
                    src_labels.append(label)
                    trg_samples.append(noisy_spec)
                    trg_labels.append(label)
                    total_clips += 1

            except Exception as e:
                print(f"    [Error] Skipping {path}: {e}")

    if total_clips == 0:
        print("[Error] No audio clips processed! Please ensure data/audio/<genre>/*.wav files exist.")
        return

    src_samples = np.array(src_samples, dtype=np.float32)  # [N, 1, 128, 256]
    src_labels = np.array(src_labels, dtype=np.int64)
    trg_samples = np.array(trg_samples, dtype=np.float32)  # [N, 1, 128, 256]
    trg_labels = np.array(trg_labels, dtype=np.int64)

    print(f"[Dataset Preparation] Total processed clips: {len(src_samples)}")
    print(f"[Dataset Preparation] Mel-Spectrogram shape: {src_samples[0].shape}")

    # Stratified Train/Test Split (80% train, 20% test)
    indices = np.arange(len(src_samples))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=src_labels
    )

    # Package Source (Clean)
    train_clean = {"samples": torch.from_numpy(src_samples[train_idx]), "labels": torch.from_numpy(src_labels[train_idx])}
    test_clean = {"samples": torch.from_numpy(src_samples[test_idx]), "labels": torch.from_numpy(src_labels[test_idx])}

    # Package Target (Noisy)
    train_noisy = {"samples": torch.from_numpy(trg_samples[train_idx]), "labels": torch.from_numpy(trg_labels[train_idx])}
    test_noisy = {"samples": torch.from_numpy(trg_samples[test_idx]), "labels": torch.from_numpy(trg_labels[test_idx])}

    torch.save(train_clean, os.path.join(output_dir, "train_clean.pt"))
    torch.save(test_clean, os.path.join(output_dir, "test_clean.pt"))
    torch.save(train_noisy, os.path.join(output_dir, "train_noisy.pt"))
    torch.save(test_noisy, os.path.join(output_dir, "test_noisy.pt"))

    print(f"[Done] Dataset successfully saved to '{output_dir}':")
    print(f"  - train_clean.pt: {train_clean['samples'].shape}")
    print(f"  - test_clean.pt:  {test_clean['samples'].shape}")
    print(f"  - train_noisy.pt: {train_noisy['samples'].shape}")
    print(f"  - test_noisy.pt:  {test_noisy['samples'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cross-Environment Music Dataset")
    parser.add_argument("--raw_audio_dir", default="data/audio", type=str, help="Directory containing clean audio genres")
    parser.add_argument("--output_dir", default="data/MusicGenre", type=str, help="Output directory for .pt files")
    parser.add_argument("--max_files", default=None, type=int, help="Optional max files per genre for fast testing")
    args = parser.parse_args()

    prepare_dataset(
        raw_audio_dir=args.raw_audio_dir,
        output_dir=args.output_dir,
        max_files_per_genre=args.max_files
    )
