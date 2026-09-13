import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense
)


def build_bilstm(
    sequence_length=200,
    embedding_dim=200,
    lstm_units=100,
    dropout=0.2
):

    model = Sequential([
        Input(
            shape=(
                sequence_length,
                embedding_dim
            )
        ),

        Bidirectional(
            LSTM(
                lstm_units,
                dropout=dropout
            )
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )


    return model