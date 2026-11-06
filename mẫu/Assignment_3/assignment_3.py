import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.phonert import LABEL_PAD_ID, PhoNERDataset, SequenceVocab, collate_fn
from models.taggers.bilstm import BiLSTMTagger


ROOT = Path(__file__).resolve().parents[2]


def find_default_data_dir() -> Path:
    candidates = (
        ROOT / "datasets" / "PhoNER_COVID19" / "data" / "word",
        ROOT / "datasets" / "PhoNER" / "data" / "word",
        ROOT / "datasets" / "PhoNER",
        ROOT / "data" / "PhoNER_COVID19" / "data" / "word",
        ROOT / "data" / "PhoNER" / "data" / "word",
        ROOT / "data" / "PhoNER",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def bio_spans(labels: list[str]) -> set[tuple[str, int, int]]:
    """Convert BIO/BIOES labels to (entity type, start, exclusive end) spans."""
    spans: set[tuple[str, int, int]] = set()
    active_type: str | None = None
    start = 0

    def close(end: int) -> None:
        nonlocal active_type
        if active_type is not None:
            spans.add((active_type, start, end))
            active_type = None

    for index, label in enumerate(labels + ["O"]):
        if label == "O" or "-" not in label:
            close(index)
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix in {"B", "S", "U"}:
            close(index)
            active_type, start = entity_type, index
            if prefix in {"S", "U"}:
                close(index + 1)
        elif prefix in {"I", "E", "L"}:
            if active_type != entity_type:
                close(index)
                active_type, start = entity_type, index
            if prefix in {"E", "L"}:
                close(index + 1)
        else:
            close(index)
    return spans


def entity_f1(gold_sequences: list[list[str]], predicted_sequences: list[list[str]]) -> float:
    true_positive = predicted_count = gold_count = 0
    for gold, predicted in zip(gold_sequences, predicted_sequences):
        gold_spans, predicted_spans = bio_spans(gold), bio_spans(predicted)
        true_positive += len(gold_spans & predicted_spans)
        predicted_count += len(predicted_spans)
        gold_count += len(gold_spans)
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch: int) -> float:
    model.train()
    losses = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as progress:
        for batch in progress:
            logits = model(batch["input_ids"].to(device), batch["lengths"])
            labels = batch["labels"].to(device)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, dataloader, vocab, device, description: str) -> float:
    model.eval()
    gold_sequences, predicted_sequences = [], []
    for batch in tqdm(dataloader, desc=description):
        logits = model(batch["input_ids"].to(device), batch["lengths"])
        predictions = logits.argmax(dim=-1).cpu()
        for row, length in enumerate(batch["lengths"].tolist()):
            gold_sequences.append([vocab.i2l[i] for i in batch["labels"][row, :length].tolist()])
            predicted_sequences.append([vocab.i2l[i] for i in predictions[row, :length].tolist()])
    score = entity_f1(gold_sequences, predicted_sequences)
    print(f"Entity-level micro F1: {score:.4f}")
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 3: BiLSTM NER on PhoNER")
    parser.add_argument("--data-dir", type=Path, default=find_default_data_dir())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = {
        split: args.data_dir / f"{split}_word.conll" for split in ("train", "dev", "test")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing PhoNER files. Put train_word.conll, dev_word.conll and "
            "test_word.conll in --data-dir. Missing:\n" + "\n".join(missing)
        )

    vocab = SequenceVocab(paths["train"], lowercase=args.lowercase)
    datasets = {split: PhoNERDataset(path, vocab) for split, path in paths.items()}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
        )
        for split, dataset in datasets.items()
    }
    model = BiLSTMTagger(
        vocab_size=len(vocab),
        hidden_size=256,
        n_layers=5,
        n_labels=vocab.n_labels,
        padding_idx=vocab.pad_idx,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_PAD_ID)

    checkpoint = Path(__file__).with_name("best_bilstm_ner.pt")
    best_f1, epochs_without_improvement = -1.0, 0
    for epoch in range(1, args.max_epochs + 1):
        train_epoch(model, loaders["train"], loss_fn, optimizer, device, epoch)
        dev_f1 = evaluate(model, loaders["dev"], vocab, device, f"Epoch {epoch} - Validation")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_f1 = evaluate(model, loaders["test"], vocab, device, "Testing best checkpoint")
    print(f"Test entity-level micro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
