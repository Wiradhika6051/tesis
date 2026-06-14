import json

from sklearn.model_selection import (
    train_test_split
)


def create_split(
    samples,
    output_file
):

    labels = [
        x["label"]
        for x in samples
    ]

    train, temp = train_test_split(
        samples,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    temp_labels = [
        x["label"]
        for x in temp
    ]

    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    split = {
        "train": train,
        "val": val,
        "test": test
    }

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(split, f)