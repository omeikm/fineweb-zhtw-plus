"""
Download FineWeb2 (Mandarin Chinese, Han script) parquet shards from the Hugging Face Hub.

Usage:
    python download.py <LOCAL_DIR> [MAX_FILES]

Examples:
    # download everything
    python download.py data/fineweb2

    # download only the first 6 shards (handy for building a small sample set)
    python download.py data/fineweb2 6

Files land in:
    <LOCAL_DIR>/data/cmn_Hani/train/*.parquet
which is exactly what to pass as <INPUT_DIR> to transform.py:
    python transform.py data/fineweb2/data/cmn_Hani/train fineweb2_cmn data/parsed 16

Notes:
  - Interrupted downloads can be resumed: just re-run the same command.
  - If you hit rate limits or want faster downloads, log in first (`huggingface-cli login`)
    or set the HF_TOKEN environment variable. `pip install hf_transfer` and
    `export HF_HUB_ENABLE_HF_TRANSFER=1` can speed things up further.
"""
import sys

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "HuggingFaceFW/fineweb-2"
SUBSET = "cmn_Hani"  # FineWeb2 config name: <iso639-3>_<script>
SPLIT = "train"
N_WORKERS = 8  # parallel download threads


if __name__ == "__main__":
    LOCAL_DIR = sys.argv[1]
    MAX_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # List the parquet shards of this subset/split (sorted so "first N" is deterministic)
    prefix = f"data/{SUBSET}/{SPLIT}/"
    files = sorted(
        f
        for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not files:
        raise SystemExit(f"No parquet files found under {prefix} in {REPO_ID}")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]
    print(f"Downloading {len(files)} files to {LOCAL_DIR}/{prefix}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=files,  # exact file list; only these shards are fetched
        local_dir=LOCAL_DIR,
        max_workers=N_WORKERS,
    )
    print("Done.")