import os
import torch
import pandas as pd
import lmdb
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

class EmotionDataset(Dataset):
    def __init__(self, csv_path, lmdb_path):
        self.df = pd.read_csv(csv_path)
        self.lmdb_path = lmdb_path
        self._env = None  # Lazy open
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["angry", "happy", "sad", "neutral"])
        self.df["label"] = self.label_encoder.transform(self.df["label"].str.lower())
        self.labels = self.df["label"].tolist()
        self.segment_ids = self.df["segment_id"].tolist()

    def __len__(self):
        return len(self.df)

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
        return self._env

    def __getitem__(self, idx):
        env = self._get_env()
        segment_id_str = self.segment_ids[idx].strip()
        segment_id = segment_id_str.encode()

        with env.begin() as txn:
            value = txn.get(segment_id)

            if value is None:
                print(f"[❌ LMDB 누락] segment_id '{segment_id_str}' (인덱스 {idx})")
                raise KeyError(f"[❌ LMDB 누락] segment_id '{segment_id_str}' 를 찾을 수 없습니다.")

            data = pickle.loads(value)

        audio = data["audio_seq"]
        text = data["text_seq"]
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        if text.dim() == 3:
            text = text.squeeze(0)

        return audio, text, self.labels[idx]

def collate_fn_padded(batch):
    audio_seqs, text_seqs, labels = zip(*batch)
    audio_padded = pad_sequence(audio_seqs, batch_first=True)
    text_padded = pad_sequence(text_seqs, batch_first=True)
    labels = torch.tensor(labels)

    audio_lengths = [a.shape[0] for a in audio_seqs]
    text_lengths = [t.shape[0] for t in text_seqs]
    max_audio_len = audio_padded.size(1)
    max_text_len = text_padded.size(1)

    audio_mask = torch.tensor([[False]*l + [True]*(max_audio_len - l) for l in audio_lengths])
    text_mask = torch.tensor([[False]*l + [True]*(max_text_len - l) for l in text_lengths])

    return audio_padded, text_padded, audio_mask, text_mask, labels
