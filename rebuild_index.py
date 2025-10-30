#!/usr/bin/env python3
"""
Rebuild FAISS index from DuckDB embeddings.

This script:
1. Loads all embeddings from the DuckDB database
2. Builds a FAISS index for fast similarity search
3. Saves the index and note ID mapping to disk

Run this script whenever:
- New notes are added to the database
- Embeddings are updated
- You want to change filtering criteria (e.g., only index helpful notes)
"""

import duckdb
import faiss
import numpy as np
import pickle
import time
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DB_PATH = str(SCRIPT_DIR / "community_notes.db")
INDEX_PATH = str(SCRIPT_DIR / "notes_index.faiss")
MAPPING_PATH = str(SCRIPT_DIR / "note_ids_mapping.pkl")

def build_faiss_index(filter_helpful_only=False):
    """
    Build FAISS index from database embeddings.

    Args:
        filter_helpful_only: If True, only index notes marked as helpful
    """
    print("=" * 60)
    print("REBUILDING FAISS INDEX")
    print("=" * 60)

    # Connect to database
    print("\n1. Connecting to database...")
    start = time.time()
    con = duckdb.connect(DB_PATH, read_only=True)
    print(f"   ✓ Connected in {time.time() - start:.3f}s")

    # Load embeddings
    print("\n2. Loading embeddings from database...")
    start = time.time()

    if filter_helpful_only:
        print("   (Filtering for CURRENTLY_RATED_HELPFUL notes only)")
        query = """
            SELECT DISTINCT n.noteId, n.summary_embedding
            FROM notes n
            LEFT JOIN note_status_history s ON n.noteId = s.noteId
            WHERE n.summary_embedding IS NOT NULL
              AND s.currentStatus = 'CURRENTLY_RATED_HELPFUL'
        """
    else:
        query = """
            SELECT noteId, summary_embedding
            FROM notes
            WHERE summary_embedding IS NOT NULL
        """

    rows = con.execute(query).fetchall()
    elapsed = time.time() - start
    print(f"   ✓ Loaded {len(rows):,} embeddings in {elapsed:.3f}s")

    if len(rows) == 0:
        print("\n❌ ERROR: No embeddings found in database!")
        print("   Run generate_embeddings.py first to create embeddings.")
        con.close()
        return False

    # Extract note IDs and embeddings
    print("\n3. Preparing data for FAISS...")
    start = time.time()
    note_ids = [row[0] for row in rows]
    embeddings = np.array([row[1] for row in rows], dtype=np.float32)
    print(f"   ✓ Prepared {len(note_ids):,} embeddings")
    print(f"   ✓ Shape: {embeddings.shape}")
    print(f"   ✓ Time: {time.time() - start:.3f}s")

    # Build FAISS index
    print("\n4. Building FAISS index...")
    start = time.time()

    # Get embedding dimension
    dimension = embeddings.shape[1]
    print(f"   Dimension: {dimension}")

    # Use IndexFlatIP for exact cosine similarity search
    # (Inner Product is equivalent to cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(dimension)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add embeddings to index
    index.add(embeddings)

    elapsed = time.time() - start
    print(f"   ✓ Index built in {elapsed:.3f}s")
    print(f"   ✓ Total vectors in index: {index.ntotal:,}")

    # Save FAISS index
    print("\n5. Saving FAISS index to disk...")
    start = time.time()
    faiss.write_index(index, INDEX_PATH)
    elapsed = time.time() - start
    print(f"   ✓ Index saved to: {INDEX_PATH}")
    print(f"   ✓ Time: {elapsed:.3f}s")

    # Save note ID mapping
    print("\n6. Saving note ID mapping...")
    start = time.time()
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(note_ids, f)
    elapsed = time.time() - start
    print(f"   ✓ Mapping saved to: {MAPPING_PATH}")
    print(f"   ✓ Time: {elapsed:.3f}s")

    # Get file sizes
    index_size_mb = Path(INDEX_PATH).stat().st_size / (1024**2)
    mapping_size_mb = Path(MAPPING_PATH).stat().st_size / (1024**2)

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Total embeddings indexed: {len(note_ids):,}")
    print(f"Index file size: {index_size_mb:.2f} MB")
    print(f"Mapping file size: {mapping_size_mb:.2f} MB")
    print(f"Total size: {index_size_mb + mapping_size_mb:.2f} MB")
    print("\nThe MCP server will now load this index on startup for fast searches.")

    con.close()
    return True

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index from Community Notes database"
    )
    parser.add_argument(
        "--helpful-only",
        action="store_true",
        help="Only index notes marked as CURRENTLY_RATED_HELPFUL"
    )

    args = parser.parse_args()

    try:
        success = build_faiss_index(filter_helpful_only=args.helpful_only)
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
