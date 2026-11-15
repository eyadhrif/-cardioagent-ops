"""TP3 step 1: turn every .wav of ourdataset/train into rows of 30 audio features.

One recording -> several 5 s segments -> one row of features per segment.
Run:  uv run python -m audio_engine.features [--rebuild]
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SAMPLE_RATE = 2000          # Hz: heart sounds are below ~1000 Hz, so 2000 Hz is enough (Nyquist)
SEGMENT_SECONDS = 5.0       # length of one segment
MIN_LAST_SECONDS = 2.0      # the last piece is kept (zero-padded) only if it is at least 2 s long
N_MFCC = 13
N_FFT = 256                 # window of 256 samples = 128 ms at 2000 Hz
HOP_LENGTH = 64             # step between two windows = 32 ms
N_MELS = 40

LABELS = {"normal": 0, "abnormal": 1}

# Paths come from environment variables (Docker sets them), with local defaults.
DATA_DIR = Path(os.getenv("DATA_DIR", "ourdataset/train"))
FEATURES_CSV = Path(os.getenv("FEATURES_CSV", "ourdataset/features.csv"))


def feature_names() -> list[str]:
    """The 30 column names, in the same order as segment_features()."""
    names = [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    names += [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
    names += ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
    return names


def load_audio(path: Path) -> np.ndarray:
    """Load a wav: resample to 2000 Hz, mono, then peak-normalise to [-1, 1]."""
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak > 0:  # avoid dividing by zero on a silent file
        signal = signal / peak
    return signal.astype(np.float32)


def split_segments(signal: np.ndarray) -> list[np.ndarray]:
    """Cut the signal into non-overlapping 5 s segments (each exactly 10000 samples)."""
    seg_len = int(SEGMENT_SECONDS * SAMPLE_RATE)
    min_last = int(MIN_LAST_SECONDS * SAMPLE_RATE)

    # Rule 1: a recording shorter than 5 s becomes ONE zero-padded segment.
    if len(signal) < seg_len:
        return [np.pad(signal, (0, seg_len - len(signal)))]

    segments = []
    for start in range(0, len(signal), seg_len):
        piece = signal[start:start + seg_len]
        if len(piece) == seg_len:
            segments.append(piece)
        elif len(piece) >= min_last:
            # Rule 2: last piece >= 2 s -> zero-pad it to 5 s
            segments.append(np.pad(piece, (0, seg_len - len(piece))))
        # Rule 3: last piece < 2 s -> dropped (mostly silence after padding)
    return segments


def segment_features(segment: np.ndarray) -> np.ndarray:
    """30 numbers that describe one segment."""
    # MFCC: 13 coefficients per frame -> shape (13, n_frames)
    mfcc = librosa.feature.mfcc(
        y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    # ZCR: how often the signal crosses 0 (noisy / high-frequency sounds -> high ZCR)
    zcr = librosa.feature.zero_crossing_rate(segment, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    # RMS: energy (loudness) of each frame
    rms = librosa.feature.rms(y=segment, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    # Summarise each frame-by-frame curve by its mean and standard deviation over time.
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),   # 13 + 13
        [zcr.mean(), zcr.std()],               # 2
        [rms.mean(), rms.std()],               # 2
    ])


def extract_file(path: Path, label: int, file_id: str) -> list[dict]:
    """All rows (one per segment) for one recording."""
    rows = []
    for i, segment in enumerate(split_segments(load_audio(path))):
        row = {"file": file_id, "segment": i, "label": label}
        row.update(zip(feature_names(), segment_features(segment)))
        rows.append(row)
    return rows


def build_dataset(data_dir: Path = DATA_DIR, cache: Path = FEATURES_CSV, rebuild: bool = False) -> pd.DataFrame:
    """Read the cache if it exists, otherwise extract the features of every wav and save them."""
    if cache.exists() and not rebuild:
        print(f"Loading cached features from {cache}")
        return pd.read_csv(cache)

    rows = []
    for class_name, label in LABELS.items():
        wav_files = sorted((data_dir / class_name).glob("*.wav"))
        print(f"Extracting {class_name}: {len(wav_files)} files")
        for n, path in enumerate(wav_files, start=1):
            # file id = "normal/a0001.wav": unique even if two classes contain the same file name
            rows.extend(extract_file(path, label, f"{class_name}/{path.name}"))
            if n % 500 == 0:
                print(f"  {n}/{len(wav_files)}")

    df = pd.DataFrame(rows, columns=["file", "segment", "label", *feature_names()])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Saved {len(df)} segments from {df['file'].nunique()} files to {cache}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MFCC/ZCR/RMS features (TP3)")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache and extract again")
    args = parser.parse_args()
    build_dataset(rebuild=args.rebuild)
