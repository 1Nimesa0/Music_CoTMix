"""
Environmental Noise Simulation and Acoustic Augmentation Module.

Simulates realistic cross-environment acoustic perturbations:
1. Calibrated SNR mixing with colored noise (Pink, Brown, Babble, Traffic)
2. Synthetic Room Impulse Response (RIR) reverberation convolving (RT60: 0.2s - 0.8s)
3. Microphone frequency response & band-pass distortion (phone/edge mic modeling)
"""

import numpy as np
import scipy.signal


def generate_colored_noise(num_samples, noise_type="pink"):
    """
    Generates colored noise based on 1/f^alpha power spectral density.
    - White: alpha = 0 (flat spectrum)
    - Pink: alpha = 1 (1/f, natural environmental noise, babble, rain)
    - Brown: alpha = 2 (1/f^2, traffic rumble, engine hum, low frequency vibration)
    """
    white = np.random.randn(num_samples)
    X = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(num_samples)
    frequencies[0] = 1e-6  # avoid division by zero

    if noise_type == "pink":
        S = 1.0 / np.sqrt(frequencies)
    elif noise_type == "brown":
        S = 1.0 / frequencies
    else:  # white
        S = np.ones_like(frequencies)

    X = X * S
    noise = np.fft.irfft(X, n=num_samples)
    # Normalize to zero mean, unit variance
    noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-8)
    return noise.astype(np.float32)


def generate_synthetic_rir(sr=16000, rt60=0.4):
    """
    Generates a synthetic Room Impulse Response (RIR) filter using an
    exponentially decaying gaussian noise model (Schroeder reverb model).
    
    Args:
        sr: Sample rate in Hz.
        rt60: Reverberation time in seconds (time to decay by 60 dB).
    Returns:
        1D impulse response array.
    """
    rir_length = int(rt60 * sr)
    if rir_length < 1:
        return np.array([1.0], dtype=np.float32)
    
    t = np.arange(rir_length) / sr
    # Exponential decay rate: -6.91 / rt60 corresponds to -60 dB at t = rt60
    decay = np.exp(-6.91 * t / max(rt60, 0.05))
    rir = np.random.randn(rir_length) * decay
    # Ensure direct path impulse at t=0
    rir[0] = 1.0
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir.astype(np.float32)


def apply_reverberation(audio, sr=16000, rt60=0.4):
    """Convolves audio with synthetic RIR to simulate room acoustics."""
    rir = generate_synthetic_rir(sr, rt60)
    reverbed = scipy.signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    return reverbed.astype(np.float32)


def apply_microphone_filter(audio, sr=16000, low_cut=100.0, high_cut=5500.0):
    """
    Simulates consumer smartphone / edge microphone frequency constraints
    using a 4th-order Butterworth bandpass filter.
    """
    nyquist = 0.5 * sr
    low = max(low_cut / nyquist, 0.01)
    high = min(high_cut / nyquist, 0.99)
    if low >= high:
        return audio
    
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    filtered = scipy.signal.filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def mix_noise_snr(clean_audio, noise, snr_db):
    """
    Mixes clean audio and noise according to a designated Signal-to-Noise Ratio (SNR) in dB.
    
    Formula:
        SNR_dB = 10 * log10( P_signal / P_noise )
        alpha = sqrt( P_signal / (P_noise * 10^(SNR/10)) )
        mixed = clean + alpha * noise
    """
    clean_audio = clean_audio.astype(np.float32)
    noise = noise.astype(np.float32)

    # Align lengths
    if len(noise) < len(clean_audio):
        repeats = int(np.ceil(len(clean_audio) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean_audio)]
    else:
        noise = noise[:len(clean_audio)]

    p_clean = np.mean(clean_audio ** 2)
    p_noise = np.mean(noise ** 2)

    if p_clean < 1e-10:
        return clean_audio

    snr_linear = 10.0 ** (snr_db / 10.0)
    alpha = np.sqrt(p_clean / (p_noise * snr_linear + 1e-10))
    mixed = clean_audio + alpha * noise

    # Peak normalization to prevent digital clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed.astype(np.float32)


def simulate_environmental_audio(clean_audio, sr=16000, profile="random", snr_db=None):
    """
    End-to-end acoustic simulation pipeline converting clean audio into
    cross-environment noisy audio.
    
    Profiles:
        - 'cafe': Pink noise chatter + short reverb (RT60 ~ 0.35s) + SNR 5-10 dB
        - 'street': Brown noise traffic rumble + open air (RT60 ~ 0.15s) + SNR 0-8 dB
        - 'car': Low-frequency Brown vibration + heavy bass rumble + SNR 2-8 dB
        - 'hall': Moderate pink noise + large reverberation (RT60 ~ 0.70s) + SNR 8-15 dB
        - 'random': Randomly samples acoustic parameters
    """
    if profile == "random":
        profile = np.random.choice(["cafe", "street", "car", "hall"])

    if profile == "cafe":
        noise_type = "pink"
        rt60 = np.random.uniform(0.3, 0.45)
        snr = np.random.uniform(5.0, 10.0) if snr_db is None else snr_db
        apply_mic = True
    elif profile == "street":
        noise_type = "brown"
        rt60 = np.random.uniform(0.1, 0.25)
        snr = np.random.uniform(0.0, 8.0) if snr_db is None else snr_db
        apply_mic = True
    elif profile == "car":
        noise_type = "brown"
        rt60 = np.random.uniform(0.15, 0.3)
        snr = np.random.uniform(2.0, 8.0) if snr_db is None else snr_db
        apply_mic = False
    elif profile == "hall":
        noise_type = "pink"
        rt60 = np.random.uniform(0.55, 0.8)
        snr = np.random.uniform(8.0, 15.0) if snr_db is None else snr_db
        apply_mic = False
    else:
        noise_type = "pink"
        rt60 = 0.3
        snr = 10.0 if snr_db is None else snr_db
        apply_mic = False

    # 1. Apply Room Impulse Response Reverberation
    reverbed = apply_reverberation(clean_audio, sr=sr, rt60=rt60)

    # 2. Add Ambient Environmental Colored Noise
    noise = generate_colored_noise(len(reverbed), noise_type=noise_type)
    noisy = mix_noise_snr(reverbed, noise, snr_db=snr)

    # 3. Simulate Smartphone Microphone Frequency Distortion
    if apply_mic:
        noisy = apply_microphone_filter(noisy, sr=sr)

    return noisy
