import numpy as np
import torch
import torch.utils.data
import librosa

MAX_WAV_VALUE = 32768.0


def wav_to_mel(
    wav: np.array,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    window_length: int = 1024,
    window: str = "hann",
    center: bool = False,
    pad_mode: str = "pad",
    num_mel_bins: int = 80,
):
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        n_mels=num_mel_bins,
    )
    return mel_spec


def mel_to_wav(
    mel_spec,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    window: str = "hann",
    center: bool = False,
    pad_mode: str = "pad",
):
    wav = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    return wav


def mel_to_logmel(mel_spec: torch.Tensor):
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def logmel_to_mel(log_spec: torch.Tensor):
    log_spec = log_spec * 4.0 - 4.0
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel_spec = torch.pow(10, log_spec).clamp(min=1e-10)
    return mel_spec
