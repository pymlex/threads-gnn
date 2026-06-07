import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm


SNAP_URL = (
    "https://snap.stanford.edu/data/reddit_threads.zip"
)


def download_dataset(raw_dir: str | Path) -> Path:
    """Download and extract the SNAP Reddit Threads dataset."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "reddit_threads.zip"
    target_csv = raw_dir / "reddit_target.csv"
    target_json = raw_dir / "reddit_edges.json"

    if target_csv.exists() and target_json.exists():
        return raw_dir

    if not zip_path.exists():
        response = requests.get(SNAP_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with zip_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Downloading reddit_threads.zip",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                progress.update(len(chunk))

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)

    nested_dir = raw_dir / "reddit_threads"
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            destination = raw_dir / item.name
            if not destination.exists():
                item.rename(destination)

    return raw_dir
