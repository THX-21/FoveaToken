from typing import List

import numpy as np
from librosa import resample


def downsample_audio(audio_array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform from its original sample rate to a target rate.

    Args:
        audio_array: Waveform samples. The caller is responsible for passing
            the shape expected by `librosa.resample` for the model path using it.
        original_sr: Sample rate of `audio_array`, in Hz.
        target_sr: Desired output sample rate, in Hz.

    Returns:
        The same audio content resampled to `target_sr`.
    """

    audio_resample_array = resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resample_array


def split_audio(audio_arrays: np.ndarray, chunk_lim: int) -> List:
    """Split a waveform into consecutive fixed-length sample chunks.

    The last chunk is kept even when it is shorter than `chunk_lim`; no padding
    is added here.

    Args:
        audio_arrays: Waveform samples to split.
        chunk_lim: Maximum number of samples per returned chunk.

    Returns:
        A list of views/slices over `audio_arrays`, each at most `chunk_lim`
        samples long.
    """

    audio_splits = []
    # Split the loaded audio to 30s chunks and extend the messages content
    for i in range(
        0,
        len(audio_arrays),
        chunk_lim,
    ):
        audio_splits.append(audio_arrays[i : i + chunk_lim])
    return audio_splits
