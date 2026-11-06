from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


LABEL_PAD_ID = -100


def read_conll(path: str | Path) -> list[tuple[list[str], list[str]]]:
    """Read a CoNLL file whose first and last columns are token and BIO tag."""
    examples: list[tuple[list[str], list[str]]] = []
    tokens: list[str] = []
    labels: list[str] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                if tokens:
                    examples.append((tokens, labels))
                    tokens, labels = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            columns = line.split()
            if len(columns) < 2:
                raise ValueError(f"Invalid CoNLL row at {path}:{line_number}: {line!r}")
            tokens.append(columns[0])
            labels.append(columns[-1])
    if tokens:
        examples.append((tokens, labels))
    if not examples:
        raise ValueError(f"No sentences found in {path}")
    return examples


class SequenceVocab:
    def __init__(self, train_path: str | Path, lowercase: bool = False):
        self.pad, self.unk = "<pad>", "<unk>"
        self.lowercase = lowercase
        examples = read_conll(train_path)
        words = {self.normalize(token) for tokens, _ in examples for token in tokens}
        labels = {label for _, sequence_labels in examples for label in sequence_labels}

        self.w2i = {self.pad: 0, self.unk: 1}
        self.w2i.update({word: i for i, word in enumerate(sorted(words), start=2)})
        self.l2i = {label: i for i, label in enumerate(sorted(labels))}
        self.i2l = {i: label for label, i in self.l2i.items()}

    def normalize(self, token: str) -> str:
        return token.lower() if self.lowercase else token

    def __len__(self) -> int:
        return len(self.w2i)

    @property
    def n_labels(self) -> int:
        return len(self.l2i)

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def encode_tokens(self, tokens: Sequence[str]) -> torch.Tensor:
        unk_idx = self.w2i[self.unk]
        return torch.tensor(
            [self.w2i.get(self.normalize(token), unk_idx) for token in tokens],
            dtype=torch.long,
        )

    def encode_labels(self, labels: Sequence[str]) -> torch.Tensor:
        unknown = set(labels).difference(self.l2i)
        if unknown:
            raise ValueError(f"Labels not present in training set: {sorted(unknown)}")
        return torch.tensor([self.l2i[label] for label in labels], dtype=torch.long)


class PhoNERDataset(Dataset):
    def __init__(self, path: str | Path, vocab: SequenceVocab):
        self.examples = read_conll(path)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens, labels = self.examples[idx]
        return {
            "input_ids": self.vocab.encode_tokens(tokens),
            "labels": self.vocab.encode_labels(labels),
        }


def collate_fn(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(item["input_ids"]) for item in items], dtype=torch.long)
    max_len = int(lengths.max())
    input_ids = torch.stack(
        [F.pad(item["input_ids"], (0, max_len - len(item["input_ids"])), value=0) for item in items]
    )
    labels = torch.stack(
        [F.pad(item["labels"], (0, max_len - len(item["labels"])), value=LABEL_PAD_ID) for item in items]
    )
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}
