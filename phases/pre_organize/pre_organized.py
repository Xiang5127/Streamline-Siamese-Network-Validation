import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "raw_datasets_terracepacels"
IMAGES_PER_HOUSE = 5
FOLDER_PREFIX = "siamese_terrace_"


def organize():
    # Collect top-level jpg files only (ignore any existing subfolders)
    jpg_files = sorted(
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".jpg"
    )

    total = len(jpg_files)
    print(f"Found {total} top-level .jpg files in '{SOURCE_DIR.name}/'")

    if total == 0:
        print("Nothing to do.")
        return

    if total % IMAGES_PER_HOUSE != 0:
        print(
            f"WARNING: {total} images is not divisible by {IMAGES_PER_HOUSE}. "
            f"The last folder will have {total % IMAGES_PER_HOUSE} image(s)."
        )

    house_index = 1
    for chunk_start in range(0, total, IMAGES_PER_HOUSE):
        chunk = jpg_files[chunk_start: chunk_start + IMAGES_PER_HOUSE]
        dest_folder = SOURCE_DIR / f"{FOLDER_PREFIX}{house_index}"
        dest_folder.mkdir(exist_ok=True)

        moved = []
        for img_path in chunk:
            dest_file = dest_folder / img_path.name
            if dest_file.exists():
                moved.append(f"  [skipped, already exists] {img_path.name}")
            else:
                shutil.move(str(img_path), str(dest_file))
                moved.append(f"  {img_path.name}")

        print(f"\n{dest_folder.name}/")
        for entry in moved:
            print(entry)

        house_index += 1

    total_folders = house_index - 1
    print(f"\nDone. Created/updated {total_folders} folder(s) under '{SOURCE_DIR.name}/'.")


if __name__ == "__main__":
    organize()
