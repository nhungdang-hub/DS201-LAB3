import json
import string
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def collate_fn(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length feedback sentences."""
    input_ids = [item["input_ids"] for item in items]
    lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())
    padded_ids = torch.stack(
        [F.pad(ids, (0, max_len - len(ids)), value=0) for ids in input_ids]
    )
    labels = torch.stack([item["label"] for item in items])
    return {"input_ids": padded_ids, "lengths": lengths, "label": labels}


class Vocab:
    def __init__(self, train_path: str | Path, label_key: str = "topic"):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.label_key = label_key

        with Path(train_path).open(encoding="utf-8") as file:
            data = json.load(file)

        words: set[str] = set()
        labels: set[str] = set()
        for item in data:
            words.update(self.preprocess_sentence(item["sentence"]).split())
            labels.add(item[label_key])

        self.w2i = {self.pad: 0, self.unk: 1}
        self.w2i.update({word: i for i, word in enumerate(sorted(words), start=2)})
        self.i2w = {i: word for word, i in self.w2i.items()}
        self.l2i = {label: i for i, label in enumerate(sorted(labels))}
        self.i2l = {i: label for label, i in self.l2i.items()}

    @property
    def n_labels(self) -> int:
        return len(self.l2i)

    def __len__(self) -> int:
        return len(self.w2i)

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    @staticmethod
    def preprocess_sentence(sentence: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return sentence.lower().translate(translator)

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        words = self.preprocess_sentence(sentence).split() or [self.unk]
        unk_idx = self.w2i[self.unk]
        return torch.tensor(
            [self.w2i.get(word, unk_idx) for word in words], dtype=torch.long
        )

    def encode_label(self, label: str) -> torch.Tensor:
        return torch.tensor(self.l2i[label], dtype=torch.long)

    def decode_labels(self, label_ids: torch.Tensor) -> list[str]:
        return [self.i2l[idx] for idx in label_ids.tolist()]


class UITVSFC(Dataset):
    def __init__(self, path: str | Path, vocab: Vocab):
        with Path(path).open(encoding="utf-8") as file:
            self.data = json.load(file)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "input_ids": self.vocab.encode_sentence(item["sentence"]),
            "label": self.vocab.encode_label(item[self.vocab.label_key]),
        }
