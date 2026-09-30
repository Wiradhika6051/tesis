import os
import sys
import json
import numpy as np

from gensim.models import Word2Vec


# ============================================================
# Configuration
# ============================================================

INPUT_MODEL = "../data/w2v/word2vec_withString10-300-200.model"

OUTPUT_FILE = "../data/w2v/vudenc_word2vec_300_200.npz"

EXPECTED_DIMENSION = 200


# ============================================================
# Load original VUDENC Word2Vec model
# ============================================================

print("=" * 80)
print("VUDENC WORD2VEC EXPORT")
print("=" * 80)

print()
print("[INFO] Python:", sys.version)
print("[INFO] NumPy:", np.__version__)

try:
    import gensim

    print("[INFO] Gensim:", gensim.__version__)

except Exception as exc:

    print("[ERROR] Could not determine Gensim version.")
    print(exc)
    sys.exit(1)


print()
print("[INFO] Loading:")
print("       {}".format(INPUT_MODEL))

if not os.path.exists(INPUT_MODEL):

    print()
    print("[ERROR] Model file does not exist:")
    print("       {}".format(INPUT_MODEL))

    sys.exit(1)


model = Word2Vec.load(
    INPUT_MODEL
)

print("[OK] Model loaded.")


# ============================================================
# Inspect model
# ============================================================

word_vectors = model.wv

vector_size = word_vectors.vector_size
vocabulary_size = len(word_vectors.vocab)

print()
print("[INFO] Vector size     :", vector_size)
print("[INFO] Vocabulary size :", vocabulary_size)


if vector_size != EXPECTED_DIMENSION:

    raise ValueError(
        "Expected {}-dimensional vectors, got {}.".format(
            EXPECTED_DIMENSION,
            vector_size
        )
    )


# ============================================================
# Extract vocabulary
# ============================================================

print()
print("[INFO] Extracting vocabulary...")

#
# Gensim 3.x stores vocabulary order in index2word.
#
tokens = list(
    word_vectors.index2word
)

print("[OK] Extracted {} tokens.".format(
    len(tokens)
))


# ============================================================
# Extract vectors
# ============================================================

print()
print("[INFO] Extracting vectors...")

vectors = np.asarray(
    word_vectors.vectors,
    dtype=np.float32
)

print("[OK] Vector matrix shape:",
      vectors.shape)


if vectors.shape != (
    len(tokens),
    EXPECTED_DIMENSION
):

    raise ValueError(
        "Unexpected vector matrix shape: {}".format(
            vectors.shape
        )
    )


# ============================================================
# Verify token/vector alignment
# ============================================================

print()
print("[INFO] Verifying token/vector alignment...")

for index in range(
    min(10, len(tokens))
):

    token = tokens[index]

    original_vector = word_vectors[token]

    exported_vector = vectors[index]

    if not np.allclose(
        original_vector,
        exported_vector
    ):

        raise ValueError(
            "Vector mismatch at index {}: {}".format(
                index,
                token
            )
        )


print("[OK] Token/vector alignment verified.")


# ============================================================
# Create output directory
# ============================================================

output_directory = os.path.dirname(
    OUTPUT_FILE
)

if output_directory and not os.path.exists(
    output_directory
):

    os.makedirs(
        output_directory
    )


# ============================================================
# Export
# ============================================================

print()
print("[INFO] Saving:")
print("       {}".format(OUTPUT_FILE))

np.savez_compressed(
    OUTPUT_FILE,
    tokens=np.asarray(
        tokens,
        dtype=object
    ),
    vectors=vectors
)

print("[OK] Export completed.")


# ============================================================
# Save metadata
# ============================================================

metadata_file = (
    OUTPUT_FILE
    + ".json"
)

metadata = {

    "source_model": INPUT_MODEL,

    "vector_size": int(
        vector_size
    ),

    "vocabulary_size": int(
        vocabulary_size
    ),

    "dtype": "float32",

    "format": "NumPy NPZ",

    "gensim_version": gensim.__version__,

    "numpy_version": np.__version__

}


with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print()
print("[OK] Metadata saved:")
print("     {}".format(metadata_file))


# ============================================================
# Final verification
# ============================================================

print()
print("[INFO] Reloading exported file...")

exported = np.load(
    OUTPUT_FILE,
    allow_pickle=True
)

loaded_tokens = exported[
    "tokens"
]

loaded_vectors = exported[
    "vectors"
]

print(
    "[INFO] Loaded tokens :",
    loaded_tokens.shape
)

print(
    "[INFO] Loaded vectors:",
    loaded_vectors.shape
)


if len(loaded_tokens) != vocabulary_size:

    raise ValueError(
        "Vocabulary size changed after export."
    )


if loaded_vectors.shape != (
    vocabulary_size,
    EXPECTED_DIMENSION
):

    raise ValueError(
        "Vector shape changed after export."
    )


#
# Verify several actual vectors after saving/reloading.
#
test_indices = [
    0,
    vocabulary_size // 2,
    vocabulary_size - 1
]

for index in test_indices:

    original = vectors[index]

    reloaded = loaded_vectors[index]

    if not np.allclose(
        original,
        reloaded
    ):

        raise ValueError(
            "Reloaded vector mismatch at index {}.".format(
                index
            )
        )


print()
print("=" * 80)
print("EXPORT SUCCESSFUL")
print("=" * 80)

print()
print("Vocabulary :", vocabulary_size)
print("Dimension  :", vector_size)
print("Output     :", OUTPUT_FILE)
print()