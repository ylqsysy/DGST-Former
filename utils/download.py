import hashlib
import time
import urllib.request

import numpy as np

from .common import resolve_path, save_json, sha256

DATASETS = {
    "PEMS03": {"shape": [26208, 358, 1], "md5": "651add9bb9eaf7f5eda2f2ee8778a182", "reported_range": "2018-09-01 to 2018-11-30"},
    "PEMS04": {"shape": [16992, 307, 3], "md5": "5238832691101da85fcac772a32051da", "reported_range": "2018-01-01 to 2018-02-28"},
    "PEMS07": {"shape": [28224, 883, 1], "md5": "978d3d9b85fe640a446983a34271a48d", "reported_range": "2017-05 to 2017-08 (published range; timestamp mapping unavailable)"},
    "PEMS08": {"shape": [17856, 170, 3], "md5": "2a528d169c0d90294c9e24288a430132", "reported_range": "2016-07-01 to 2016-08-31"},
}
BASE = "https://zenodo.org/records/7816008/files/"
EXPECTED_SMALL = {
    "PEMS03.csv": "e6cf2ef1e1a9957aeec678a30b7ce7a7",
    "PEMS03.txt": "a52b864fbbe9d852ebdd08f6191a987f",
    "PEMS04.csv": "3990731ec980019a875a6d5723523855",
    "PEMS07.csv": "cea609fed853ade176cb1c5eae0aeda0",
    "PEMS08.csv": "5e507b2b18d9235e95b63a20221cd4c5",
}


def fetch_file(url, path, retries=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DGST-Former/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, open(pending, "wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
            pending.replace(path)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def download_dataset(name, data_root="data"):
    spec = DATASETS[name]
    base = resolve_path(data_root)
    folder = base / "raw" / name
    records = []
    for suffix in ["npz", "csv"] + (["txt"] if name == "PEMS03" else []):
        filename = f"{name}.{suffix}"
        path = folder / filename
        url = BASE + filename + "?download=1"
        if not path.exists():
            fetch_file(url, path)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        expected = spec["md5"] if suffix == "npz" else EXPECTED_SMALL[filename]
        if digest != expected:
            raise ValueError(f"{filename}: checksum mismatch; remove the invalid file and retry")
        if suffix == "npz":
            with np.load(path, allow_pickle=False) as archive:
                if list(archive["data"].shape) != spec["shape"]:
                    raise ValueError(f"{name}: nonstandard dataset shape")
        records.append({"file": filename, "source": url, "sha256": sha256(path), "md5": digest})
    metadata = {"dataset": name, "archive_doi": "10.5281/zenodo.7816008", "files": records, **spec}
    save_json(base / "metadata" / f"{name}_source.json", metadata)
    return metadata
