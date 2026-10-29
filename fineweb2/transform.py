"""
Convert HF FineWeb2 parquet shards into the layout/schema that 1_filter_lang.py expects,
so that stages 0_prefilter and 0b_trafilatura can be skipped.

1_filter_lang.py reads:
    {MAIN_OUTPUT_PATH}/{DUMP}/0b_traf/output/*.parquet
with datatrove's ParquetReader (default adapter), which maps:
    text      -> Document.text
    id        -> Document.id
    media     -> Document.media     (optional, defaults to [])
    metadata  -> Document.metadata  (everything else ends up here)

FineWeb2 shards have flat columns:
    text, id, dump, url, date, file_path, language, language_score,
    language_script, minhash_cluster_size, top_langs, ...
This script keeps `text` and `id` as top-level columns and packs all the other
columns into a single `metadata` struct column (same shape as datatrove's ParquetWriter output).

Usage:
    python transform.py <INPUT_DIR> <DUMP> <MAIN_OUTPUT_PATH> <N_CPU> [CC_DUMP_FILTER]

Example:
    python transform.py data/fineweb2/cmn_Hani/train fineweb2_cmn data/parsed 16
    # optional: keep only one Common Crawl snapshot (matches FineWeb2's `dump` column)
    python transform.py data/fineweb2/cmn_Hani/train fineweb2_cmn data/parsed 16 CC-MAIN-2024-26

Output:
    data/parsed/fineweb2_cmn/0b_traf/output/<same file name as input>.parquet
Then run:
    python 1_filter_lang.py fineweb2_cmn data/parsed 16
"""
import os
import sys
import glob
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH_SIZE = 10_000  # rows per batch; keeps memory bounded for big shards
TOP_LEVEL_COLS = ("text", "id")  # columns kept at the top level


def convert_table(table: pa.Table, cc_dump_filter: str | None) -> pa.Table | None:
    """Convert one FineWeb2 record batch (as a table) to the 0b_traf schema."""
    # Optionally keep only one Common Crawl snapshot
    if cc_dump_filter and "dump" in table.column_names:
        table = table.filter(pc.equal(table["dump"], cc_dump_filter))

    # Drop rows with empty / null text (same behaviour as the has_text filter in 0b)
    mask = pc.and_(
        pc.is_valid(table["text"]),
        pc.greater(pc.utf8_length(pc.fill_null(table["text"], "")), 0),
    )
    table = table.filter(mask)
    if table.num_rows == 0:
        return None

    # Every column that is not text/id goes into the `metadata` struct
    meta_names = [c for c in table.column_names if c not in TOP_LEVEL_COLS]
    metadata = pa.StructArray.from_arrays(
        [table[c].combine_chunks() for c in meta_names],
        names=meta_names,
    )

    return pa.table(
        {
            "text": table["text"],
            "id": table["id"].cast(pa.string()),
            "metadata": metadata,
        }
    )


def process_file(args):
    in_path, out_path, cc_dump_filter = args

    # Skip files that were already converted (same resume behaviour as 0b)
    if os.path.exists(out_path):
        return in_path, "skipped", 0

    # Write to a temp file first so an interrupted run never leaves a half-written parquet
    tmp_path = out_path + ".tmp"
    writer = None
    n_rows = 0
    try:
        with pq.ParquetFile(in_path) as pf:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE):
                out = convert_table(pa.Table.from_batches([batch]), cc_dump_filter)
                if out is None:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, out.schema, compression="snappy")
                writer.write_table(out)
                n_rows += out.num_rows
    except Exception as e:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return in_path, f"error: {e}", 0

    if writer is None:
        # Nothing survived the filters; do not create an empty output file
        return in_path, "empty", 0

    writer.close()
    os.replace(tmp_path, out_path)
    return in_path, "ok", n_rows


if __name__ == "__main__":
    INPUT_DIR = sys.argv[1]
    DUMP = sys.argv[2]  # only used as the folder name in the output tree
    MAIN_OUTPUT_PATH = sys.argv[3]
    N_CPU = int(sys.argv[4])
    CC_DUMP_FILTER = sys.argv[5] if len(sys.argv) > 5 else None

    # Must match MAIN_INPUT_PATH_LAST_STAGE in 1_filter_lang.py
    out_dir = os.path.join(MAIN_OUTPUT_PATH, DUMP, "0b_traf", "output")
    os.makedirs(out_dir, exist_ok=True)

    in_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not in_files:
        raise SystemExit(f"No parquet files found in {INPUT_DIR}")

    jobs = [
        (f, os.path.join(out_dir, os.path.basename(f)), CC_DUMP_FILTER)
        for f in in_files
    ]
    print(f"Converting {len(jobs)} files -> {out_dir}")

    total = 0
    with Pool(min(N_CPU, len(jobs))) as pool:
        for path, status, n in pool.imap_unordered(process_file, jobs):
            total += n
            print(f"[{status}] {os.path.basename(path)} ({n} rows)")

    print(f"Done. {total} documents written.")