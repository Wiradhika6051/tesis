import os
import json
import random

import numpy as np

from gensim.models import Word2Vec

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential
from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

SEED = 42

MAX_LENGTH = 200
EMBEDDING_DIM = 200

LSTM_UNITS = 100
DROPOUT = 0.2

BATCH_SIZE = 128
EPOCHS = 100

W2V_PATH = "w2v/word2vec_withString10-300-200.model"

MODEL_DIR = "model"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Word2Vec
# ============================================================

print("=" * 60)
print("Loading Word2Vec")
print("=" * 60)

w2v_model = Word2Vec.load(W2V_PATH)
word_vectors = w2v_model.wv

print("Vocabulary size :", len(word_vectors))
print("Embedding dim   :", word_vectors.vector_size)


# ============================================================
# Build embedding matrix
# ============================================================

def build_embedding_matrix(word_vectors):

    vocab_size = len(word_vectors.key_to_index)

    embedding_matrix = np.zeros(
        (vocab_size + 1, word_vectors.vector_size),
        dtype=np.float32
    )

    token_to_id = {}

    for idx, token in enumerate(
        word_vectors.key_to_index.keys(),
        start=1
    ):

        token_to_id[token] = idx

        embedding_matrix[idx] = (
            word_vectors[token]
        )

    return token_to_id, embedding_matrix


token_to_id, embedding_matrix = (
    build_embedding_matrix(word_vectors)
)

VOCAB_SIZE = embedding_matrix.shape[0]

print("Vocabulary size :", VOCAB_SIZE)
print("Embedding shape :", embedding_matrix.shape)


# ============================================================
# CFG → sequence
# ============================================================

def get_pruned_code(sample):
    """
    Convert the pruned CFG into a sequential representation.

    Nodes are ordered by source line number.
    Each node contributes its source text.
    """

    nodes = sample.pruned_cfg["nodes"]

    #
    # Sort nodes according to source order.
    #
    nodes = sorted(
        nodes,
        key=lambda node: (
            node.lineno,
            node.node_id
        )
    )

    texts = []

    for node in nodes:

        text = node.text

        if text is None:
            continue

        text = str(text).strip()

        if text:
            texts.append(text)

    #
    # Concatenate pruned CFG nodes.
    #
    return "\n".join(texts)


# ============================================================
# Tokenization
# ============================================================

def tokenize_code(code):
    """
    Tokenize source code.

    IMPORTANT:
    Replace this implementation with the tokenizer used
    by your existing pipeline / VUDENC if exact tokenization
    is required.
    """

    #
    # Temporary simple tokenizer.
    #
    import re

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|"
        r"\d+(?:\.\d+)?|"
        r"==|!=|<=|>=|"
        r"->|"
        r"[^\s]",
        code
    )


# ============================================================
# Encode sample
# ============================================================

def encode_sample(sample):

    code = get_pruned_code(sample)

    tokens = tokenize_code(code)

    ids = []

    for token in tokens:

        token_id = token_to_id.get(
            token,
            0
        )

        ids.append(token_id)

    return ids


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(samples):

    X = []
    y = []

    skipped = 0

    for sample in samples:

        if sample.pruned_cfg is None:
            skipped += 1
            continue

        nodes = sample.pruned_cfg.get(
            "nodes",
            []
        )

        if len(nodes) == 0:
            skipped += 1
            continue

        sequence = encode_sample(
            sample
        )

        if len(sequence) == 0:
            skipped += 1
            continue

        X.append(sequence)

        #
        # Your current dataset convention:
        # 1 = vulnerable
        # 0 = clean
        #
        y.append(
            int(sample.label)
        )

    X = pad_sequences(
        X,
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre",
        value=0
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    print(
        f"Prepared {len(X)} samples "
        f"(skipped {skipped})"
    )

    return X, y

def build_model():

    model = Sequential()

    model.add(
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LENGTH,
            trainable=False
        )
    )

    model.add(
        Bidirectional(
            LSTM(
                LSTM_UNITS,
                dropout=DROPOUT,
                recurrent_dropout=DROPOUT
            )
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy"
        ]
    )

    return model

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    probabilities = model.predict(
        X,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print()
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(
        f"Samples   : {len(y)}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"FPR       : {fpr:.4f}"
    )

    print()
    print("Confusion Matrix")
    print(cm)

    print(
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )

    return {
        "dataset": dataset_name,
        "samples": len(y),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def train_baseline(
    train_samples,
    val_samples,
    test_samples,
    pruner_name
):

    print()
    print("=" * 60)
    print(
        f"Training BiLSTM: {pruner_name}"
    )
    print("=" * 60)

    #
    # Prepare data
    #
    X_train, y_train = prepare_dataset(
        train_samples
    )

    X_val, y_val = prepare_dataset(
        val_samples
    )

    X_test, y_test = prepare_dataset(
        test_samples
    )

    print()
    print("Shapes")
    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Val  :",
        X_val.shape,
        y_val.shape
    )

    print(
        "Test :",
        X_test.shape,
        y_test.shape
    )

    #
    # Class weights
    #
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight
        in zip(classes, weights)
    }

    print()
    print(
        "Class weights:",
        class_weights
    )

    #
    # Build model
    #
    model = build_model()

    model.summary()

    #
    # Early stopping
    #
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    #
    # Train
    #
    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    #
    # Evaluation
    #
    train_result = evaluate_model(
        model,
        X_train,
        y_train,
        "Train"
    )

    val_result = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation"
    )

    test_result = evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    #
    # Save model
    #
    model_path = os.path.join(
        MODEL_DIR,
        f"BiLSTM_{pruner_name}.h5"
    )

    model.save(
        model_path
    )

    #
    # Save results
    #
    result = {
        "pruner": pruner_name,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "bidirectional": True,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "train": train_result,
        "validation": val_result,
        "test": test_result
    }

    result_path = os.path.join(
        RESULT_DIR,
        f"BiLSTM_{pruner_name}.json"
    )

    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    return model, result