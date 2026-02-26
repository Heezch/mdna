#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve


EXPECTED_SUBDIRS = {"0_s1s1", "1_s2s2", "FI"}
SHA256_PLACEHOLDERS = {"<SHA256>", "SHA256", "YOUR_SHA256", "OPTIONAL_SHA256"}
FIGSHARE_DIRECT_URL_PATTERN = re.compile(r"https://figshare\.com/ndownloader/files/\d+")


def _compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_archive_name(url: str) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name
    if candidate:
        return candidate
    return "filament_dataset_archive.zip"


def _resolve_download_url(url: str) -> str:
    if FIGSHARE_DIRECT_URL_PATTERN.search(url):
        return url

    try:
        with urlopen(url) as response:
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
            if "text/html" not in content_type.lower():
                return final_url
            html = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return url

    match = FIGSHARE_DIRECT_URL_PATTERN.search(html)
    if match:
        resolved = match.group(0)
        print(f"Resolved Figshare file download URL: {resolved}")
        return resolved

    return url


def _looks_like_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    present = {child.name for child in path.iterdir() if child.is_dir()}
    return EXPECTED_SUBDIRS.issubset(present)


def _find_dataset_dir(root: Path) -> Path | None:
    if _looks_like_dataset_dir(root):
        return root
    for child in root.iterdir():
        if _looks_like_dataset_dir(child):
            return child
    return None


def _normalize_expected_sha256(expected_sha256: str | None) -> str | None:
    if not expected_sha256:
        return None
    normalized = expected_sha256.strip()
    normalized_upper = normalized.upper()
    if normalized in SHA256_PLACEHOLDERS or (normalized.startswith("<") and normalized.endswith(">")):
        print(
            "Ignoring placeholder checksum value. "
            "Set MDNA_FILAMENT_DATASET_SHA256 to a real hash to enable integrity verification."
        )
        return None
    return normalized


def _unpack_archive_robust(archive_path: Path, extract_dir: Path) -> None:
    try:
        shutil.unpack_archive(str(archive_path), str(extract_dir))
        return
    except shutil.ReadError:
        pass

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zip_handle:
            zip_handle.extractall(extract_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar_handle:
            tar_handle.extractall(extract_dir)
        return

    raise shutil.ReadError(f"Unknown archive format '{archive_path}'")


def fetch_dataset(
    url: str,
    output_root: Path,
    dataset_dir_name: str,
    expected_sha256: str | None,
    force: bool,
) -> Path:
    expected_sha256 = _normalize_expected_sha256(expected_sha256)
    url = _resolve_download_url(url)

    destination = output_root / dataset_dir_name
    if destination.exists():
        if not force:
            print(f"Dataset already exists at {destination}. Use --force to re-download.")
            return destination
        shutil.rmtree(destination)

    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_name = _resolve_archive_name(url)
        archive_path = tmp_path / archive_name

        print(f"Downloading from: {url}")
        urlretrieve(url, archive_path)

        if expected_sha256:
            checksum = _compute_sha256(archive_path)
            if checksum.lower() != expected_sha256.lower():
                raise ValueError(
                    "Checksum mismatch for downloaded archive. "
                    f"Expected {expected_sha256}, got {checksum}."
                )
            print("Checksum verified.")

        extracted_root = tmp_path / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        _unpack_archive_robust(archive_path, extracted_root)

        dataset_source = _find_dataset_dir(extracted_root)
        if dataset_source is None:
            raise RuntimeError(
                "Could not locate extracted filament dataset folder. "
                "Expected subdirectories: 0_s1s1, 1_s2s2, FI."
            )

        shutil.move(str(dataset_source), str(destination))

    print(f"Dataset ready at: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and install optional filament_dataset from Figshare (or any archive URL)."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("MDNA_FILAMENT_DATASET_URL"),
        help=(
            "Dataset URL. Can be a direct archive URL, a Figshare item URL, or a DOI URL. "
            "Figshare page/DOI URLs are resolved to a direct file download automatically. "
            "Can also be set via MDNA_FILAMENT_DATASET_URL."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("examples/data"),
        help="Root folder where filament_dataset will be placed.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="filament_dataset",
        help="Name of destination dataset folder.",
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=os.environ.get("MDNA_FILAMENT_DATASET_SHA256"),
        help="Optional archive SHA256 checksum. Can also be set via MDNA_FILAMENT_DATASET_SHA256.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if destination dataset folder already exists.",
    )

    args = parser.parse_args()

    if not args.url:
        raise SystemExit(
            "No dataset URL provided. Use --url <FIGSHARE_ARCHIVE_URL> or set MDNA_FILAMENT_DATASET_URL."
        )

    fetch_dataset(
        url=args.url,
        output_root=args.output_root,
        dataset_dir_name=args.dataset_dir,
        expected_sha256=args.sha256,
        force=args.force,
    )


if __name__ == "__main__":
    main()
