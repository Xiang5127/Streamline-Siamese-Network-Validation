import csv
import random
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "raw_datasets_terracepacels"
FOLDER_PREFIX = "siamese_terrace_"
SEED = 42
NUM_VAL_HOUSES = 10
TRAIN_CSV = PROJECT_ROOT / "pairs_train.csv"
VAL_CSV = PROJECT_ROOT / "pairs_val.csv"


def collect_houses():
    """Return dict: {house_id: [relative_image_paths...]} sorted by house index."""
    houses = {}
    for folder in sorted(SOURCE_DIR.iterdir(), key=lambda p: int(p.name.split("_")[-1]) if p.is_dir() and p.name.startswith(FOLDER_PREFIX) else -1):
        if not folder.is_dir() or not folder.name.startswith(FOLDER_PREFIX):
            continue
        images = sorted(folder.glob("*.jpg"))
        if not images:
            continue
        house_id = folder.name
        rel_paths = [str(img.relative_to(PROJECT_ROOT)).replace("\\", "/") for img in images]
        houses[house_id] = rel_paths
    return houses


def make_positive_pairs(houses, house_ids):
    """All C(n,2) unordered pairs per house."""
    pairs = []
    for hid in house_ids:
        imgs = houses[hid]
        for a, b in combinations(imgs, 2):
            pairs.append((a, b, 1))
    return pairs


def make_negative_pairs(houses, house_ids, count, rng):
    """Sample `count` unique cross-house pairs."""
    seen = set()
    pairs = []
    pool = list(house_ids)
    attempts = 0
    max_attempts = count * 50
    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        h1, h2 = rng.sample(pool, 2)
        a = rng.choice(houses[h1])
        b = rng.choice(houses[h2])
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b, 0))
    if len(pairs) < count:
        raise RuntimeError(
            f"Could only sample {len(pairs)}/{count} unique negative pairs after {attempts} attempts."
        )
    return pairs


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_a", "path_b", "label"])
        writer.writerows(rows)


def summarize(name, rows):
    pos = sum(1 for r in rows if r[2] == 1)
    neg = sum(1 for r in rows if r[2] == 0)
    print(f"  {name}: total={len(rows)}  positives={pos}  negatives={neg}")


def main():
    rng = random.Random(SEED)

    houses = collect_houses()
    print(f"Found {len(houses)} house folders under '{SOURCE_DIR.name}/'.")
    if not houses:
        raise SystemExit("No houses found. Run the pre-organize step first.")

    house_ids = list(houses.keys())
    rng.shuffle(house_ids)

    val_ids = sorted(house_ids[:NUM_VAL_HOUSES], key=lambda s: int(s.split("_")[-1]))
    train_ids = sorted(house_ids[NUM_VAL_HOUSES:], key=lambda s: int(s.split("_")[-1]))
    print(f"Split: {len(train_ids)} train houses / {len(val_ids)} val houses")
    print(f"  val houses: {val_ids}")

    # Train
    train_pos = make_positive_pairs(houses, train_ids)
    train_neg = make_negative_pairs(houses, train_ids, count=len(train_pos), rng=rng)
    train_rows = train_pos + train_neg
    rng.shuffle(train_rows)

    # Val
    val_pos = make_positive_pairs(houses, val_ids)
    val_neg = make_negative_pairs(houses, val_ids, count=len(val_pos), rng=rng)
    val_rows = val_pos + val_neg
    rng.shuffle(val_rows)

    write_csv(TRAIN_CSV, train_rows)
    write_csv(VAL_CSV, val_rows)

    print("\nWrote:")
    print(f"  {TRAIN_CSV.name}")
    print(f"  {VAL_CSV.name}")
    print("\nSummary:")
    summarize("train", train_rows)
    summarize("val", val_rows)

    print("\nSample rows (train):")
    for row in train_rows[:5]:
        print(f"  {row}")


if __name__ == "__main__":
    main()
