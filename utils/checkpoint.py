import hashlib
import torch


def calculate_dataset_hash(
    dataset_file
):
    """
    Generate md5 hash
    for reproducibility.
    """

    with open(
        dataset_file,
        "rb"
    ) as f:

        return hashlib.md5(
            f.read()
        ).hexdigest()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    checkpoint_file,
    split_file,
    vocab_file,
    dataset_file,
    config
):
    """
    Save training checkpoint.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "split_file":
            split_file,

        "vocab_file":
            vocab_file,

        "dataset_hash":
            calculate_dataset_hash(
                dataset_file
            ),

        "config":
            config
    }

    torch.save(
        checkpoint,
        checkpoint_file
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_file}"
    )


def load_checkpoint(
    checkpoint_file,
    model,
    optimizer=None
):
    """
    Restore model state.
    """

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        optimizer is not None
        and
        "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    print(
        f"Loaded checkpoint "
        f"from epoch "
        f"{checkpoint['epoch']}"
    )

    return checkpoint