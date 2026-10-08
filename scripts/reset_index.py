"""Clear all vectors in the configured Pinecone namespace and forget ingest progress.

Usage: python scripts/reset_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone  # noqa: E402

from app.config import get_settings  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    try:
        Pinecone(api_key=s.pinecone_api_key).Index(s.pinecone_index).delete(
            delete_all=True, namespace=s.namespace
        )
        print(f"Cleared namespace '{s.namespace}' in index '{s.pinecone_index}'.")
    except Exception as e:
        print(f"Nothing to clear ({type(e).__name__}: {e})")
    manifest = Path(s.manifest_path)
    if manifest.exists():
        manifest.unlink()
        print("Removed local ingest manifest.")
