from collections import defaultdict

from sklearn.model_selection import train_test_split


def get_pair_id(sample):
    """
    Vulnerable and fixed samples belonging to the same
    change share this identifier.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path
    )


def group_by_pair(samples):

    pairs = defaultdict(list)

    for sample in samples:

        pair_id = get_pair_id(sample)

        pairs[pair_id].append(sample)

    return pairs


def paired_split(
    samples,
    test_size=0.30,
    val_size=0.50,
    random_state=42,
    shuffle=True
):
    """
    Split the dataset by vulnerable/fixed pair.

    A vulnerable/fixed pair can never be separated
    between train, validation, and test.

    Returns:

        paired_train
        paired_val
        paired_test
        incomplete_samples
    """

    pairs = group_by_pair(
        samples
    )

    complete_pairs = []
    incomplete_samples = []

    for pair_id, pair_samples in pairs.items():

        labels = {
            sample.label
            for sample in pair_samples
        }

        #
        # A complete pair must contain:
        #
        #   label 0 = fixed
        #   label 1 = vulnerable
        #
        if labels == {0, 1}:

            complete_pairs.append(
                pair_samples
            )

        else:

            incomplete_samples.extend(
                pair_samples
            )

    #
    # One label per pair for stratification.
    #
    # Every complete pair contains one vulnerable
    # sample, so all complete pairs have the same
    # pair-level class composition.
    #
    train_pairs, temp_pairs = train_test_split(

        complete_pairs,

        test_size=test_size,

        random_state=random_state,

        shuffle=shuffle
    )

    val_pairs, test_pairs = train_test_split(

        temp_pairs,

        test_size=val_size,

        random_state=random_state,

        shuffle=shuffle
    )

    def flatten(pairs):

        return [

            sample

            for pair in pairs

            for sample in pair

        ]

    paired_train = flatten(
        train_pairs
    )

    paired_val = flatten(
        val_pairs
    )

    paired_test = flatten(
        test_pairs
    )

    return (
        paired_train,
        paired_val,
        paired_test,
        incomplete_samples
    )


def check_pair_split(
    train_set,
    val_set,
    test_set
):
    """
    Verify that no pair appears in more than one split.
    """

    partitions = {

        "train": train_set,

        "validation": val_set,

        "test": test_set

    }

    pair_locations = defaultdict(set)

    for partition, dataset in partitions.items():

        for sample in dataset:

            pair_id = get_pair_id(
                sample
            )

            pair_locations[pair_id].add(
                partition
            )

    split_pairs = {

        pair_id: locations

        for pair_id, locations
        in pair_locations.items()

        if len(locations) > 1
    }

    return split_pairs