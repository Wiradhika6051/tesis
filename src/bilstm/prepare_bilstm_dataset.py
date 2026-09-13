import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split

from src.loader import load_samples
from src.bilstm.bilstm_preprocessor import VUDENCVectorizer
from src.pipeline import DatasetPreparationPipeline
from src.cfg.cfg_builder import CFGBuilder
from src.localizer import (
    GitDiffLocalizer,
    LineCFGLocalizer,
    FunctionCFGLocalizer,
)
from src.pruners.backward_slice_pruner import BackwardSlicePruner
from src.pruners.forward_slice_pruner import ForwardSlicePruner
from src.pruners.identity_pruner import IdentityPruner
from src.pruners.neighborhood_pruner import NeighborhoodPruner


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# CHANGE THIS to your actual dataset JSONL path.
JSONL_PATH = "data/plain_sql.jsonl"

CHECKPOINT_DIR = "data/checkpoints"

W2V_PATH = "data/w2v/word2vec_withString10-300-200.model"

OUTPUT_DIR = "data/bilstm_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

samples = load_samples(
    jsonl_path=JSONL_PATH,
    checkpoint_dir=CHECKPOINT_DIR,
)

print("Total samples:", len(samples))


# ============================================================
# CREATE THE 70/15/15 SPLIT
# ============================================================

indices = np.arange(len(samples))

labels = np.asarray([
    int(sample.label)
    for sample in samples
])

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_idx],
)

base_train = [samples[i] for i in train_idx]
base_val = [samples[i] for i in val_idx]
base_test = [samples[i] for i in test_idx]

print("Train:", len(base_train))
print("Val:  ", len(base_val))
print("Test: ", len(base_test))


# ============================================================
# LOAD VUDENC WORD2VEC
# ============================================================

vectorizer = VUDENCVectorizer(W2V_PATH)


# ============================================================
# PRUNERS
# ============================================================

PRUNERS = {
    "identity": IdentityPruner(),
    "backward": BackwardSlicePruner(),
    "forward": ForwardSlicePruner(),
    "neighborhood4": NeighborhoodPruner(hops=4),
}


# ============================================================
# PREPARE AND SAVE EACH DATASET
# ============================================================

for pruner_name, pruner in PRUNERS.items():

    print()
    print("=" * 60)
    print("Processing:", pruner_name)
    print("=" * 60)

    # Every pruner starts from the SAME raw split.
    train_samples = copy.deepcopy(base_train)
    val_samples = copy.deepcopy(base_val)
    test_samples = copy.deepcopy(base_test)

    # Same preprocessing pipeline as the GCN experiment.
    pipeline = DatasetPreparationPipeline(
        cfg_builder=CFGBuilder(),
        diff_localizer=GitDiffLocalizer(),
        line_localizer=LineCFGLocalizer(),
        function_localizer=FunctionCFGLocalizer(),
        pruner=pruner,
    )

    # Creates sample.pruned_cfg.
    train_samples = pipeline.prepare(train_samples)
    val_samples = pipeline.prepare(val_samples)
    test_samples = pipeline.prepare(test_samples)

    # Converts pruned CFG nodes -> tokens -> VUDENC Word2Vec vectors.
    X_train, y_train, skipped_train = vectorizer.prepare_dataset(
        train_samples
    )

    X_val, y_val, skipped_val = vectorizer.prepare_dataset(
        val_samples
    )

    X_test, y_test, skipped_test = vectorizer.prepare_dataset(
        test_samples
    )

    print("Train:", X_train.shape, "skipped:", skipped_train)
    print("Val:  ", X_val.shape, "skipped:", skipped_val)
    print("Test: ", X_test.shape, "skipped:", skipped_test)

    output_path = os.path.join(
        OUTPUT_DIR,
        pruner_name + ".npz",
    )

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print("Saved:", output_path)


print()
print("=" * 60)
print("LOCAL PREPROCESSING COMPLETE")
print("=" * 60)
print("Expected output:")
print("  bilstm_data/identity.npz")
print("  bilstm_data/backward.npz")
print("  bilstm_data/forward.npz")
print("  bilstm_data/neighborhood4.npz")
