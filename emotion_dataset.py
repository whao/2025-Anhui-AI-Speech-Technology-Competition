import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
label_map = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

class EmotionFeatureDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=128, max_len=300,
                 split='train', train_ratio=0.8, random_state=42,
                 cache_dir='feature_cache'):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.split = split
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        all_filepaths, all_labels = [], []

        for label in EMOTION_LABELS:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for filename in os.listdir(label_dir):
                if filename.endswith('.wav'):
                    all_filepaths.append(os.path.join(label_dir, filename))
                    all_labels.append(label_map[label])

        train_files, val_files, train_labels, val_labels = train_test_split(
            all_filepaths, all_labels,
            test_size=(1 - train_ratio),
            stratify=all_labels,
            random_state=random_state
        )

        if split == 'train':
            self.filepaths, self.labels = train_files, train_labels
        elif split == 'val':
            self.filepaths, self.labels = val_files, val_labels
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        # 缓存路径（每个 .npy 文件名唯一）
        fname = os.path.basename(path).replace('.wav', '.npy')
        cache_path = os.path.join(self.cache_dir, fname)

        if os.path.exists(cache_path):
            features = np.load(cache_path)
        else:
            features = self.extract_and_save(path, cache_path)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def extract_and_save(self, path, save_path):
        y, _ = librosa.load(path, sr=self.sr)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel)

        f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=self.sr)
        f0 = np.nan_to_num(f0, nan=0.0)

        T = mel_db.shape[1]
        if len(f0) != T:
            f0 = np.interp(np.linspace(0, len(f0), T), np.arange(len(f0)), f0)
        f0 = f0[np.newaxis, :]

        features = np.vstack([mel_db, f0])
        features = self._fix_length(features, self.max_len)

        np.save(save_path, features.astype(np.float32))
        return features

    def _fix_length(self, feat, length):
        c, t = feat.shape
        if t < length:
            pad = np.zeros((c, length - t), dtype=np.float32)
            feat = np.concatenate((feat, pad), axis=1)
        elif t > length:
            feat = feat[:, :length]
        return feat
