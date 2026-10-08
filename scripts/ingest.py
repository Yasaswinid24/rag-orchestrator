"""Bulk-ingest every supported file in a directory: python scripts/ingest.py [dir]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_directory  # noqa: E402

if __name__ == "__main__":
    results = ingest_directory(sys.argv[1] if len(sys.argv) > 1 else None)
    ok = {k: v for k, v in results.items() if v > 0}
    empty = [k for k, v in results.items() if v == 0]
    failed = [k for k, v in results.items() if v < 0]
    print(f"\nIndexed {len(ok)} documents, {sum(ok.values())} chunks total.")
    if empty:
        print(f"No extractable text ({len(empty)}): {', '.join(empty)}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")