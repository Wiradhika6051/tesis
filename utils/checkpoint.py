import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    path
):

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer":
            optimizer.state_dict()
    }, path)


def load_checkpoint(
    model,
    optimizer,
    path
):

    ckpt = torch.load(path)

    model.load_state_dict(
        ckpt["model"]
    )

    optimizer.load_state_dict(
        ckpt["optimizer"]
    )

    return ckpt["epoch"]