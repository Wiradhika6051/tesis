from collections import Counter


def get_statistics(samples):
    labels = [
        getattr(sample, "label", None)
        for sample in samples
    ]

    positive = labels.count(1)
    negative = labels.count(0)

    return {
        "total": len(samples),
        "positive": positive,
        "negative": negative,
    }


def print_statistics(stage, samples):
    stats = get_statistics(samples)

    print()
    print("=" * 70)
    print("PIPELINE STAGE:", stage)
    print("=" * 70)
    print("Total    :", stats["total"])
    print("Positive :", stats["positive"])
    print("Negative :", stats["negative"])

    if stats["total"]:
        print(
            "Positive %: {:.2f}%".format(
                100.0 * stats["positive"] / stats["total"]
            )
        )

        print(
            "Negative %: {:.2f}%".format(
                100.0 * stats["negative"] / stats["total"]
            )
        )

    return stats