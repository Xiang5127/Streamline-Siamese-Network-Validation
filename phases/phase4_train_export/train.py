import tensorflow as tf
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "phases" / "phase2_data_pipeline"))
sys.path.append(str(PROJECT_ROOT / "phases" / "phase3_siamese_model"))

from data_pipeline import build_dataset
from siamese_model import build_siamese_model

ROOT = Path(__file__).parent

# ─── Hyperparameters ────────────────────────────────────────────────────────────
MARGIN = 1.0
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 16
TFLITE_OUTPUT = PROJECT_ROOT / "building_dna_extractor.tflite"
TRAIN_CSV = PROJECT_ROOT / "pairs_train.csv"
VAL_CSV = PROJECT_ROOT / "pairs_val.csv"


# ─── Custom Contrastive Loss ────────────────────────────────────────────────────
def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss for Siamese networks.

    y_true = 1 → same house (positive pair)  → penalise large distance
    y_true = 0 → different house (negative)  → penalise distance < margin
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    positive_term = y_true * tf.square(y_pred)
    negative_term = (1.0 - y_true) * tf.square(tf.maximum(MARGIN - y_pred, 0.0))
    return tf.reduce_mean(positive_term + negative_term)


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    # 1. Build datasets
    print("Loading datasets ...")
    train_ds = build_dataset(TRAIN_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(VAL_CSV, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Build & compile model
    print("Building Siamese model ...")
    model = build_siamese_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=contrastive_loss,
    )

    # 3. Train
    print(f"Training for {EPOCHS} epochs ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # 4. Extract the embedding sub-model
    embedding_network = model.get_layer("embedding_network")
    print(f"\nExtracted embedding_network: "
          f"input={embedding_network.input_shape} → output={embedding_network.output_shape}")

    # 5. Convert to TFLite with default (dynamic-range / 8-bit) optimizations
    print("Converting to TFLite with default optimizations ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(embedding_network)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    TFLITE_OUTPUT.write_bytes(tflite_model)
    size_kb = TFLITE_OUTPUT.stat().st_size / 1024
    print(f"\nSaved: {TFLITE_OUTPUT.name}  ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
