"""Fine-tuning loop for `DebertaModel`."""

import math
from collections.abc import Sequence

import torch
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup

from hiring_bias_xai.models import DebertaModel

WARMUP_RATIO = 0.06


def _evaluate(model: DebertaModel, texts: Sequence[str], labels: Sequence[str]) -> dict:
    logits = model.logits(texts)
    targets = torch.tensor([model.labels.index(label) for label in labels])
    loss = torch.nn.functional.cross_entropy(torch.from_numpy(logits), targets)
    predicted = [model.labels[i] for i in logits.argmax(axis=1)]
    return {
        "dev_loss": float(loss),
        "dev_macro_f1": float(f1_score(labels, predicted, average="macro")),
    }


def train(
    model: DebertaModel,
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    dev_texts: Sequence[str],
    dev_labels: Sequence[str],
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    eval_every: int,
) -> list[dict]:
    """Fine-tune `model` and return one row per evaluation on the dev bios."""
    network = model.network
    targets = torch.tensor([model.labels.index(label) for label in train_labels])
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    steps_per_epoch = len(train_texts) // batch_size
    total_steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(WARMUP_RATIO * total_steps), total_steps
    )
    rows, step, losses = [], 0, []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_texts))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            chosen = order[start : start + batch_size]
            network.train()
            inputs = model.tokenizer(
                [train_texts[i] for i in chosen], padding=True, truncation=True, return_tensors="pt"
            ).to(network.device)
            loss = torch.nn.functional.cross_entropy(
                network(**inputs).logits.float(), targets[chosen].to(network.device)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss))
            step += 1
            if step % eval_every == 0 or step == total_steps:
                rows.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": sum(losses) / len(losses),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    | _evaluate(model, dev_texts, dev_labels)
                )
                losses = []
    return rows
