"""
Generate best demo pairs from a building image dataset.

Scans subfolders (each = one building), computes embeddings via TFLite,
then selects the top positive (same building) and top negative (different
building) pairs sorted by Euclidean distance.

Outputs:
  - <output_dir>/pairs.csv        ranked pair list
  - <output_dir>/pair_<N>/ref.jpg  +  live.jpg   for each selected pair
  - <output_dir>/all_pair_images/  flat copy of all selected ref/live images

Usage:
  python generate_test_pairs.py --input siamese_raw_datasets --output demo_pairs
  python generate_test_pairs.py --input siamese_raw_datasets --top 10
"""

import argparse
import itertools
import os
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "building_dna_extractor.tflite"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ── TFLite helpers ───────────────────────────────────────────────────────────
def load_interpreter(model_path: str):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def get_embedding(interp, img_array: np.ndarray) -> np.ndarray:
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]["index"], img_array)
    interp.invoke()
    return interp.get_tensor(out[0]["index"]).flatten()


# ── Scan dataset ─────────────────────────────────────────────────────────────
def scan_folder(root: str):
    """Return {building_label: [image_paths]} grouped by subfolder."""
    groups = {}
    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue
        imgs = [
            os.path.join(subdir, f)
            for f in sorted(os.listdir(subdir))
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        if imgs:
            groups[entry] = imgs
    return groups


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate best demo pairs")
    parser.add_argument("--input", required=True, help="Root folder with building subfolders")
    parser.add_argument("--output", default="demo_pairs", help="Output directory (default: demo_pairs)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .tflite model")
    parser.add_argument("--top", type=int, default=5, help="Number of top positive AND negative pairs to select")
    args = parser.parse_args()

    # 1. Scan
    groups = scan_folder(args.input)
    print(f"Found {len(groups)} buildings, {sum(len(v) for v in groups.values())} total images.\n")

    # 2. Compute embeddings
    interp = load_interpreter(args.model)
    all_paths = []
    all_labels = []
    all_embeddings = []

    for label, paths in groups.items():
        for p in paths:
            img = Image.open(p)
            emb = get_embedding(interp, preprocess(img))
            all_paths.append(p)
            all_labels.append(label)
            all_embeddings.append(emb)
            print(f"  Embedded: {p}")

    all_embeddings = np.array(all_embeddings)
    n = len(all_paths)
    print(f"\nComputed {n} embeddings.\n")

    # 3. Compute all pairwise distances
    positives = []  # (dist, i, j)  same building
    negatives = []  # (dist, i, j)  different building

    for i, j in itertools.combinations(range(n), 2):
        dist = float(np.linalg.norm(all_embeddings[i] - all_embeddings[j]))
        entry = (dist, i, j)
        if all_labels[i] == all_labels[j]:
            positives.append(entry)
        else:
            negatives.append(entry)

    positives.sort(key=lambda x: x[0])          # best matches first (low dist)
    negatives.sort(key=lambda x: x[0], reverse=True)  # clearest rejections first (high dist)

    top_pos = positives[: args.top]
    top_neg = negatives[: args.top]

    print(f"Positive pairs found: {len(positives)}  |  selecting top {len(top_pos)}")
    print(f"Negative pairs found: {len(negatives)}  |  selecting top {len(top_neg)}\n")

    # 4. Write output
    os.makedirs(args.output, exist_ok=True)
    all_pairs_dir = os.path.join(args.output, "all_pair_images")
    os.makedirs(all_pairs_dir, exist_ok=True)
    csv_rows = ["pair,type,image_a,image_b,building_a,building_b,distance"]

    pair_num = 0
    for tag, selected in [("positive", top_pos), ("negative", top_neg)]:
        for dist, i, j in selected:
            pair_num += 1
            pair_dir = os.path.join(args.output, f"pair_{pair_num}")
            os.makedirs(pair_dir, exist_ok=True)

            ref_ext = os.path.splitext(all_paths[i])[1]
            live_ext = os.path.splitext(all_paths[j])[1]
            ref_dst = os.path.join(pair_dir, f"ref{ref_ext}")
            live_dst = os.path.join(pair_dir, f"live{live_ext}")

            shutil.copy2(all_paths[i], ref_dst)
            shutil.copy2(all_paths[j], live_dst)

            flat_ref_dst = os.path.join(all_pairs_dir, f"pair_{pair_num:03d}_ref{ref_ext}")
            flat_live_dst = os.path.join(all_pairs_dir, f"pair_{pair_num:03d}_live{live_ext}")
            shutil.copy2(all_paths[i], flat_ref_dst)
            shutil.copy2(all_paths[j], flat_live_dst)

            csv_rows.append(
                f"{pair_num},{tag},{all_paths[i]},{all_paths[j]},"
                f"{all_labels[i]},{all_labels[j]},{dist:.6f}"
            )
            print(f"  Pair {pair_num:>2} [{tag:>8}]  dist={dist:.4f}  "
                  f"{all_labels[i]} <-> {all_labels[j]}")

    csv_path = os.path.join(args.output, "pairs.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")

    print(f"\n✓ {pair_num} pairs saved to '{args.output}/'")
    print(f"✓ CSV written to '{csv_path}'")
    print(f"✓ Consolidated images saved to '{all_pairs_dir}'")


if __name__ == "__main__":
    main()
