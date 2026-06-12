import argparse
import zipfile
from pathlib import Path


def unzip_salt_dataset(zip_dir: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)

    zip_files = list(zip_dir.glob("*.zip"))

    if not zip_files:
        print(f"No zip files found in {zip_dir}")
        return

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(extract_to)

    print("All files extracted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract multiple zip files into a unified directory."
    )

    # Define arguments and automatically cast them to Path objects
    parser.add_argument(
        "--zip_dir",
        type=Path,
        required=True,
        help="Path to the directory containing the zip files.",
    )
    parser.add_argument(
        "--extract_to",
        type=Path,
        required=True,
        help="Path to the target extraction directory.",
    )

    args = parser.parse_args()

    unzip_salt_dataset(args.zip_dir, args.extract_to)
