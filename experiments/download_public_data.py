from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request
import zipfile


DATA_CENTER_TCP_URL = "https://zenodo.org/records/10894768/files/tcp-dataset.zip?download=1"


def download(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f" -> {output}")
    urllib.request.urlretrieve(url, output)
    return output


def extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    print(f"Extracted {path} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public TCP congestion datasets.")
    parser.add_argument("--dataset", choices=["datacenter_tcp", "puffer"], required=True)
    parser.add_argument("--url", help="Required for Puffer because those URLs are day-specific.")
    parser.add_argument("--external-root", default="data/external")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    external_root = Path(args.external_root)
    if args.dataset == "datacenter_tcp":
        url = DATA_CENTER_TCP_URL
    elif args.url:
        url = args.url
    else:
        raise SystemExit("--url is required for Puffer downloads.")

    suffix = ".zip" if ".zip" in url.lower() or args.dataset == "datacenter_tcp" else ".download"
    download_path = external_root / args.dataset / f"{args.dataset}{suffix}"
    downloaded = download(url, download_path)
    if not args.no_extract and downloaded.suffix.lower() == ".zip":
        extract_zip(downloaded, external_root / args.dataset)


if __name__ == "__main__":
    main()
