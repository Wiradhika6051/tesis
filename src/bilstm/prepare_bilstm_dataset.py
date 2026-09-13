import os
import copy

import numpy as np
from sklearn.model_selection import train_test_split

from loader import load_samples

from bilstm_preprocessor import (
    VUDENCVectorizer
)

from pipeline import (
    DatasetPreparationPipeline
)

from cfg import CFGBuilder

from localizer import (
    GitDiffLocalizer,
    LineCFGLocalizer,
    FunctionCFGLocalizer
)

from pruner import (
    IdentityPruner,
    BackwardSlicePruner,
    ForwardSlicePruner,
    NeighborhoodPruner
)


SEED = 42

W2V_PATH = (
    "w2v/"
    "word2vec_withString10-300-200.model"
)

OUTPUT_DIR = "bilstm_data"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load samples
# ============================================================

samples = load_samples(
    jsonl_path="data/your_dataset.jsonl",
    checkpoint_dir="checkpoints"
)


# ============================================================
# Fixed split
# ============================================================

indices = np.arange(
    len(samples)
)

labels = np.asarray([
    int(sample.label)
    for sample in samples
])


train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_idx]
)


base_train = [
    samples[i]
    for i in train_idx
]

base_val = [
    samples[i]
    for i in val_idx
]

base_test = [
    samples[i]
    for i in test_idx
]


# ============================================================
# Load VUDENC Word2Vec
# ============================================================

vectorizer = VUDENCVectorizer(
    W2V_PATH
)


# ============================================================
# Pruners
# ============================================================

PRUNERS = {
    "identity": IdentityPruner(),

    "backward": BackwardSlicePruner(),

    "forward": ForwardSlicePruner(),

    "neighborhood4":
        NeighborhoodPruner(hops=4)
}


# ============================================================
# Process each pruner
# ============================================================

for pruner_name, pruner in PRUNERS.items():

    print()
    print("=" * 60)
    print(
        "Processing:",
        pruner_name
    )
    print("=" * 60)

    #
    # IMPORTANT:
    # each pruner starts from the same raw split.
    #
    train_samples = copy.deepcopy(
        base_train
    )

    val_samples = copy.deepcopy(
        base_val
    )

    test_samples = copy.deepcopy(
        base_test
    )


    # --------------------------------------------------------
    # Same preprocessing pipeline as GCN
    # --------------------------------------------------------

    pipeline = DatasetPreparationPipeline(
        cfg_builder=CFGBuilder(),

        diff_localizer=GitDiffLocalizer(),

        line_localizer=LineCFGLocalizer(),

        function_localizer=
            FunctionCFGLocalizer(),

        pruner=pruner
    )


    train_samples = pipeline.prepare(
        train_samples
    )

    val_samples = pipeline.prepare(
        val_samples
    )

    test_samples = pipeline.prepare(
        test_samples
    )


    # --------------------------------------------------------
    # Word2Vec representation
    # --------------------------------------------------------

    X_train, y_train, skipped_train = (
        vectorizer.prepare_dataset(
            train_samples
        )
    )

    X_val, y_val, skipped_val = (
        vectorizer.prepare_dataset(
            val_samples
        )
    )

    X_test, y_test, skipped_test = (
        vectorizer.prepare_dataset(
            test_samples
        )
    )


    print(
        "Train:",
        X_train.shape
    )

    print(
        "Val:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        pruner_name + ".npz"
    )

    np.savez_compressed(
        output_path,

        X_train=X_train,
        y_train=y_train,

        X_val=X_val,
        y_val=y_val,

        X_test=X_test,
        y_test=y_test
    )

    print(
        "Saved:",
        output_path
    )