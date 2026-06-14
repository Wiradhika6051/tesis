import json

from sklearn.model_selection import train_test_split


def create_split(
    samples,
    output_file,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Create train/val/test split.

    Stores only sample IDs.

    Parameters
    ----------
    samples : List[dict]

    output_file : str

    train_ratio : float

    val_ratio : float

    test_ratio : float

    seed : int
    """

    assert (
        train_ratio +
        val_ratio +
        test_ratio
    ) == 1.0

    labels = [
        sample["label"]
        for sample in samples
    ]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed
    )

    temp_labels = [
        sample["label"]
        for sample in temp_samples
    ]

    val_size = val_ratio / (
        val_ratio + test_ratio
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=seed
    )

    split = {
        "train": [
            sample["id"]
            for sample in train_samples
        ],
        "val": [
            sample["id"]
            for sample in val_samples
        ],
        "test": [
            sample["id"]
            for sample in test_samples
        ]
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            split,
            f,
            indent=4
        )

    print(
        f"Split saved to {output_file}"
    )

    print(
        f"Train: {len(split['train'])}"
    )

    print(
        f"Val: {len(split['val'])}"
    )

    print(
        f"Test: {len(split['test'])}"
    )


def load_split(
    split_file,
    samples
):
    """
    Reconstruct train/val/test
    from sample IDs.
    """

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as f:
        split = json.load(f)

    sample_map = {
        sample["id"]: sample
        for sample in samples
    }

    train = [
        sample_map[idx]
        for idx in split["train"]
    ]

    val = [
        sample_map[idx]
        for idx in split["val"]
    ]

    test = [
        sample_map[idx]
        for idx in split["test"]
    ]

    return train, val, test