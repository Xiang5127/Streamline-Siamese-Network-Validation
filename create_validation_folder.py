from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
VAL_CSV = PROJECT_ROOT / "pairs_val.csv"
TRAIN_CSV = PROJECT_ROOT / "pairs_train.csv"
OUTPUT_DIR = PROJECT_ROOT / "validation_folder"


def normalize_rel_path(path_str: str) -> Path:
    """Normalize CSV path values that may contain '\\' or '/' separators."""
    return Path(path_str.replace("\\", "/"))


def collect_unique_images(csv_path: Path) -> set[Path]:
    df = pd.read_csv(csv_path)
    images = set()
    for col in ("path_a", "path_b"):
        for value in df[col].dropna().tolist():
            images.add(normalize_rel_path(str(value)))
    return images


def house_label(path: Path) -> str:
    # Expected shape: raw_datasets_terracepacels/siamese_terrace_x/IMG....jpg
    if len(path.parts) >= 2:
        return path.parts[1]
    return ""


def main():
    if not VAL_CSV.exists():
        raise SystemExit(f"Missing validation CSV: {VAL_CSV}")

    val_images = collect_unique_images(VAL_CSV)
    if not val_images:
        raise SystemExit("No validation images found in pairs_val.csv")

    train_images = set()
    if TRAIN_CSV.exists():
        train_images = collect_unique_images(TRAIN_CSV)

    overlap_images = val_images & train_images
    if overlap_images:
        raise RuntimeError(
            f"Found {len(overlap_images)} image(s) present in both training and validation CSVs."
        )

    val_houses = {house_label(p) for p in val_images if house_label(p)}
    train_houses = {house_label(p) for p in train_images if house_label(p)}
    overlap_houses = val_houses & train_houses
    if overlap_houses:
        raise RuntimeError(
            f"Found overlapping house labels between train/val: {sorted(overlap_houses)}"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    copied = 0
    missing = []
    for rel_path in sorted(val_images):
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            missing.append(str(rel_path))
            continue

        dst = OUTPUT_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Validation houses: {len(val_houses)}")
    print(f"Unique validation images listed in CSV: {len(val_images)}")
    print(f"Copied images to '{OUTPUT_DIR.name}/': {copied}")

    if missing:
        print(f"Missing files not copied: {len(missing)}")
        for item in missing[:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
