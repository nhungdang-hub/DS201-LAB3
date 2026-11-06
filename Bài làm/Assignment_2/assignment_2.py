import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.uit_vsfc import UITVSFC, Vocab, collate_fn
from models.classifiers.gru import GRUModel


ROOT = Path(__file__).resolve().parents[2]


def find_default_data_dir() -> Path:
    candidates = (ROOT / "datasets" / "UIT-VSFC", ROOT / "data" / "UIT-VSFC")
    return next((path for path in candidates if path.is_dir()), candidates[0])


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch: int) -> float:
    model.train()
    losses = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as progress:
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, batch["lengths"])
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, dataloader, device, description: str) -> float:
    model.eval()
    gold, predicted = [], []
    for batch in tqdm(dataloader, desc=description):
        logits = model(batch["input_ids"].to(device), batch["lengths"])
        gold.extend(batch["label"].tolist())
        predicted.extend(logits.argmax(dim=-1).cpu().tolist())
    score = f1_score(gold, predicted, average="macro", zero_division=0)
    print(f"Macro F1: {score:.4f}")
    return float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 2: GRU on UIT-VSFC")
    parser.add_argument("--data-dir", type=Path, default=find_default_data_dir())
    parser.add_argument("--label-key", choices=("topic", "sentiment"), default="topic")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = {
        split: args.data_dir / f"UIT-VSFC-{split}.json"
        for split in ("train", "dev", "test")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing UIT-VSFC files:\n" + "\n".join(missing))

    vocab = Vocab(paths["train"], label_key=args.label_key)
    datasets = {split: UITVSFC(path, vocab) for split, path in paths.items()}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
        )
        for split, dataset in datasets.items()
    }

    model = GRUModel(
        vocab_size=len(vocab),
        hidden_size=256,
        n_layers=5,
        n_labels=vocab.n_labels,
        padding_idx=vocab.pad_idx,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, epochs_without_improvement = -1.0, 0
    checkpoint = Path(__file__).with_name("best_gru.pt")
    for epoch in range(1, args.max_epochs + 1):
        train_epoch(model, loaders["train"], loss_fn, optimizer, device, epoch)
        dev_f1 = evaluate(model, loaders["dev"], device, f"Epoch {epoch} - Validation")
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
    test_f1 = evaluate(model, loaders["test"], device, "Testing best checkpoint")
    print(f"Test macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
