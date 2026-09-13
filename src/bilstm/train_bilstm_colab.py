import json
import os

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import (
    compute_class_weight
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from bilstm_model import (
    build_bilstm
)


DATA_DIR = "/content/bilstm_data"

RESULT_DIR = "/content/bilstm_results"

MODEL_DIR = "/content/bilstm_models"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# GPU
# ============================================================

print(
    "TensorFlow:",
    tf.__version__
)

print(
    "GPU:",
    tf.config.list_physical_devices(
        "GPU"
    )
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    X,
    y,
    name
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


    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).ravel()
    )


    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0
    )


    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

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

    print(
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )


    return {
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


# ============================================================
# Train one pruner
# ============================================================

def train_pruner(
    pruner_name
):

    print()
    print("#" * 60)
    print(
        "PRUNER:",
        pruner_name
    )
    print("#" * 60)


    data = np.load(
        os.path.join(
            DATA_DIR,
            pruner_name + ".npz"
        )
    )


    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]


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
    # Class weights
    # --------------------------------------------------------

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(c): float(w)
        for c, w in zip(
            classes,
            weights
        )
    }


    print(
        "Class weights:",
        class_weights
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_bilstm(
        sequence_length=200,
        embedding_dim=200,
        lstm_units=100,
        dropout=0.2
    )


    model.summary()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )


    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=100,

        batch_size=128,

        class_weight=class_weights,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    train_result = evaluate(
        model,
        X_train,
        y_train,
        "TRAIN"
    )

    val_result = evaluate(
        model,
        X_val,
        y_val,
        "VALIDATION"
    )

    test_result = evaluate(
        model,
        X_test,
        y_test,
        "TEST"
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "BiLSTM_" +
        pruner_name +
        ".keras"
    )

    model.save(
        model_path
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "model": "BiLSTM",
        "pruner": pruner_name,

        "seed": 42,

        "sequence_length": 200,

        "embedding_dimension": 200,

        "lstm_units": 100,

        "dropout": 0.2,

        "batch_size": 128,

        "max_epochs": 100,

        "train": train_result,

        "validation": val_result,

        "test": test_result
    }


    result_path = os.path.join(
        RESULT_DIR,
        "BiLSTM_" +
        pruner_name +
        ".json"
    )


    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


# ============================================================
# Run all pruners
# ============================================================

for pruner in [
    "identity",
    "backward",
    "forward",
    "neighborhood4"
]:

    train_pruner(
        pruner
    )