from pathlib import Path

import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_preprocess(path, image_size):
    """Read file -> decode JPEG -> resize -> float32 in [0,1]."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _make_load_fn(image_size):
    def _load_pair(path_a, path_b, label):
        img_a = _decode_and_preprocess(path_a, image_size)
        img_b = _decode_and_preprocess(path_b, image_size)
        return (
            {"reference_image": img_a, "live_image": img_b},
            tf.cast(label, tf.float32),
        )

    return _load_pair


def build_dataset(
    csv_path,
    batch_size=16,
    image_size=(224, 224),
    shuffle=True,
    shuffle_buffer=1024,
    seed=42,
):
    """
    Build a tf.data.Dataset for Siamese training from a pairs CSV.

    CSV schema: path_a,path_b,label (paths relative to project root).
    Yields per-batch: ({"reference_image": ..., "live_image": ...}, label)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Resolve relative paths against project root so tf.io.read_file works
    # regardless of the CWD the script is invoked from.
    path_a = [str((PROJECT_ROOT / p).resolve()) for p in df["path_a"].tolist()]
    path_b = [str((PROJECT_ROOT / p).resolve()) for p in df["path_b"].tolist()]
    labels = df["label"].astype("int32").tolist()

    ds = tf.data.Dataset.from_tensor_slices((path_a, path_b, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(shuffle_buffer, len(df)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_make_load_fn(image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if __name__ == "__main__":
    # Smoke test: build the train dataset and inspect one batch.
    train_csv = PROJECT_ROOT / "pairs_train.csv"
    if not train_csv.exists():
        raise SystemExit(
            f"Missing {train_csv.name}. Run generate_pairs.py first."
        )

    ds = build_dataset(train_csv, batch_size=16)
    print(f"Dataset built from {train_csv.name}")
    print(f"Element spec: {ds.element_spec}\n")

    for inputs, labels in ds.take(1):
        ref = inputs["reference_image"]
        live = inputs["live_image"]
        print("Batch shapes:")
        print(f"  reference_image: {ref.shape}  dtype={ref.dtype}")
        print(f"  live_image     : {live.shape}  dtype={live.dtype}")
        print(f"  label          : {labels.shape}  dtype={labels.dtype}")

        print("\nPixel range check:")
        print(f"  reference_image min={tf.reduce_min(ref).numpy():.4f}  max={tf.reduce_max(ref).numpy():.4f}")
        print(f"  live_image      min={tf.reduce_min(live).numpy():.4f}  max={tf.reduce_max(live).numpy():.4f}")

        print(f"\nLabel sample: {labels.numpy().tolist()}")
