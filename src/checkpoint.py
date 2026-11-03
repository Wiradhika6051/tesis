
"""Checkpoint serialization helpers for model training."""

import hashlib
import os

import torch


def calculate_dataset_hash(dataset_file):
    """Return the dataset's MD5 hash, or None if no path is supplied."""

    if not dataset_file:
        return None

    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(
            f"Dataset file does not exist: {dataset_file}"
        )

    digest = hashlib.md5()

    with open(dataset_file, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    checkpoint_file,
    split_file="",
    vocab_file="",
    dataset_file="",
    config="",
    trainer_state=None,
):
    """Save model, optimizer, metadata, and optional trainer state."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "split_file": split_file,
        "vocab_file": vocab_file,
        "dataset_hash": calculate_dataset_hash(dataset_file),
        "config": config,
    }

    if trainer_state:
        checkpoint.update(trainer_state)

    checkpoint_dir = os.path.dirname(
        os.fspath(checkpoint_file)
    )

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(checkpoint, checkpoint_file)

    print(f"Checkpoint saved: {checkpoint_file}")


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Restore model weights and optionally optimizer state."""

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )

    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']}"
    )

    return checkpoint
