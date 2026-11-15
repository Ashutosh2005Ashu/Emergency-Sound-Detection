from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parent
    emergency_dir: Path = root / "emergency"
    non_emergency_dir: Path = root / "non-emeregency"
    sample_rate: int = 16_000
    window_seconds: float = 1.5
    hop_seconds: float = 0.5
    random_state: int = 42


def load_audio(file_path: Path, sr: int) -> np.ndarray:
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    if signal.size == 0:
        raise ValueError(f"Audio has no samples: {file_path}")
    signal = signal / (np.max(np.abs(signal)) + 1e-12)
    return signal.astype(np.float32)


def split_into_windows(signal: np.ndarray, sr: int, window_s: float, hop_s: float) -> list[np.ndarray]:
    window_len = int(window_s * sr)
    hop_len = int(hop_s * sr)
    if window_len <= 0 or hop_len <= 0:
        raise ValueError("Window and hop sizes must be positive.")

    if signal.shape[0] < window_len:
        pad = window_len - signal.shape[0]
        return [np.pad(signal, (0, pad))]

    windows: list[np.ndarray] = []
    for start in range(0, signal.shape[0] - window_len + 1, hop_len):
        end = start + window_len
        windows.append(signal[start:end])

    if not windows:
        windows.append(signal[:window_len])
    return windows


def extract_features(window: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(window)
    rms = librosa.feature.rms(y=window)
    centroid = librosa.feature.spectral_centroid(y=window, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=window, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=window, sr=sr)

    feature_parts = [
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.array(
            [
                float(np.mean(zcr)),
                float(np.std(zcr)),
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.mean(centroid)),
                float(np.std(centroid)),
                float(np.mean(bandwidth)),
                float(np.std(bandwidth)),
                float(np.mean(rolloff)),
                float(np.std(rolloff)),
            ]
        ),
    ]
    return np.concatenate(feature_parts, axis=0)


def collect_files(folder: Path) -> list[Path]:
    extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]
    return files


def build_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray, list[str]]:
    file_groups = {
        "emergency": collect_files(cfg.emergency_dir),
        "non_emergency": collect_files(cfg.non_emergency_dir),
    }
    if not file_groups["emergency"] or not file_groups["non_emergency"]:
        raise FileNotFoundError(
            "Missing audio files. Put files in 'emergency/' and 'non-emeregency/' folders."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []

    label_map = {"non_emergency": 0, "emergency": 1}

    for class_name, files in file_groups.items():
        for audio_file in files:
            signal = load_audio(audio_file, cfg.sample_rate)
            windows = split_into_windows(signal, cfg.sample_rate, cfg.window_seconds, cfg.hop_seconds)
            for window in windows:
                features.append(extract_features(window, cfg.sample_rate))
                labels.append(label_map[class_name])
                names.append(audio_file.name)

    x = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    return x, y, names


def train_model(cfg: Config) -> Pipeline:
    x, y, _ = build_dataset(cfg)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=4.0, gamma="scale", probability=True, random_state=cfg.random_state)),
        ]
    )
    model.fit(x, y)
    return model


def predict_audio_file(cfg: Config, model: Pipeline, audio_path: Path) -> tuple[str, float]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Test file not found: {audio_path}")

    signal = load_audio(audio_path, cfg.sample_rate)
    windows = split_into_windows(signal, cfg.sample_rate, cfg.window_seconds, cfg.hop_seconds)
    features = np.vstack([extract_features(window, cfg.sample_rate) for window in windows])

    window_preds = model.predict(features)
    window_probs = model.predict_proba(features)[:, 1]
    emergency_ratio = float(np.mean(window_preds))
    avg_emergency_prob = float(np.mean(window_probs))

    final_label = "Emergency" if emergency_ratio >= 0.5 else "Non-Emergency"
    return final_label, avg_emergency_prob


def train_and_evaluate(cfg: Config) -> None:
    x, y, source_names = build_dataset(cfg)
    class_names = np.array(["non_emergency", "emergency"])

    x_train, x_test, y_train, y_test, src_train, src_test = train_test_split(
        x,
        y,
        source_names,
        test_size=0.25,
        random_state=cfg.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=4.0, gamma="scale", probability=True, random_state=cfg.random_state)),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Mini Project: Emergency Sound Detection ===")
    print(f"Total windows: {len(y)}")
    print(f"Train windows: {len(y_train)}")
    print(f"Test windows: {len(y_test)}")
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Emergency Sound Detection")
    plt.tight_layout()
    out_path = cfg.root / "confusion_matrix.png"
    plt.savefig(out_path, dpi=180)
    print(f"Saved confusion matrix: {out_path}")

    print("\nTest file windows summary:")
    for name, y_true, y_hat in zip(src_test, y_test, y_pred):
        print(f"{name:45s} true={class_names[y_true]:14s} pred={class_names[y_hat]}")


if __name__ == "__main__":
    config = Config()
    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1]).expanduser().resolve()
        clf = train_model(config)
        label, emergency_prob = predict_audio_file(config, clf, test_path)
        print(f"File: {test_path}")
        print(f"Prediction: {label}")
        print(f"Emergency probability: {emergency_prob:.4f}")
    else:
        train_and_evaluate(config)