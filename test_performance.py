#!/usr/bin/env python3
"""
Performance testing script for Community Notes MCP tools.
Tests query speed and identifies bottlenecks.
"""

import duckdb
import numpy as np
import faiss
import pickle
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Database path
script_dir = Path(__file__).parent.absolute()
db_path = str(script_dir / "community_notes.db")
index_path = str(script_dir / "notes_index.faiss")
mapping_path = str(script_dir / "note_ids_mapping.pkl")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def test_connection():
    """Test database connection and get basic stats."""
    print("=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)

    start = time.time()
    con = duckdb.connect(db_path, read_only=True)
    elapsed = time.time() - start
    print(f"✓ Database connected in {elapsed:.3f}s")
    print(f"  Path: {db_path}")

    # Get database size
    import os
    db_size_gb = os.path.getsize(db_path) / (1024**3)
    print(f"  Size: {db_size_gb:.2f} GB")

    return con

def test_simple_query(con):
    """Test simple query performance."""
    print("\n" + "=" * 60)
    print("SIMPLE QUERY TEST")
    print("=" * 60)

    # Count total notes
    start = time.time()
    total_notes = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    elapsed = time.time() - start
    print(f"✓ Total notes: {total_notes:,} (query: {elapsed:.3f}s)")

    # Count notes with embeddings
    start = time.time()
    with_embeddings = con.execute(
        "SELECT COUNT(*) FROM notes WHERE summary_embedding IS NOT NULL"
    ).fetchone()[0]
    elapsed = time.time() - start
    print(f"✓ Notes with embeddings: {with_embeddings:,} (query: {elapsed:.3f}s)")

    return with_embeddings

def test_filtered_query(con):
    """Test filtered query with status."""
    print("\n" + "=" * 60)
    print("FILTERED QUERY TEST (with JOIN)")
    print("=" * 60)

    start = time.time()
    result = con.execute("""
        SELECT
            n.noteId,
            n.tweetId,
            n.summary,
            s.currentStatus
        FROM notes n
        LEFT JOIN note_status_history s ON n.noteId = s.noteId
        WHERE s.currentStatus = 'CURRENTLY_RATED_HELPFUL'
        LIMIT 10
    """).fetchall()
    elapsed = time.time() - start

    print(f"✓ Found {len(result)} helpful notes (query: {elapsed:.3f}s)")
    return elapsed

def test_embedding_retrieval(con, limit=1000):
    """Test how long it takes to retrieve embeddings."""
    print("\n" + "=" * 60)
    print(f"EMBEDDING RETRIEVAL TEST (fetching {limit} embeddings)")
    print("=" * 60)

    start = time.time()
    result = con.execute(f"""
        SELECT noteId, summary_embedding
        FROM notes
        WHERE summary_embedding IS NOT NULL
        LIMIT {limit}
    """).fetchall()
    elapsed = time.time() - start

    print(f"✓ Retrieved {len(result)} embeddings in {elapsed:.3f}s")
    print(f"  Rate: {len(result)/elapsed:.0f} embeddings/sec")

    # Check embedding dimensions
    if result:
        emb_dims = len(result[0][1])
        print(f"  Embedding dimensions: {emb_dims}")

    return elapsed, result

def test_semantic_search_full(con, query_text="election misinformation", limit=10):
    """Test full semantic search (current approach - loads ALL embeddings)."""
    print("\n" + "=" * 60)
    print(f"SEMANTIC SEARCH TEST - FULL SCAN")
    print("=" * 60)
    print(f"Query: '{query_text}'")
    print(f"Looking for top {limit} results")

    # Load model
    print("\nLoading embedding model...")
    model_start = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model_elapsed = time.time() - model_start
    print(f"✓ Model loaded in {model_elapsed:.3f}s")

    # Generate query embedding
    print("\nGenerating query embedding...")
    emb_start = time.time()
    query_embedding = model.encode(query_text).tolist()
    emb_elapsed = time.time() - emb_start
    print(f"✓ Query embedding generated in {emb_elapsed:.3f}s")

    # Fetch ALL embeddings from database
    print("\nFetching all embeddings from database...")
    fetch_start = time.time()
    result = con.execute("""
        SELECT
            n.noteId,
            n.summary,
            n.summary_embedding
        FROM notes n
        WHERE n.summary_embedding IS NOT NULL
    """).fetchall()
    fetch_elapsed = time.time() - fetch_start
    print(f"✓ Fetched {len(result):,} embeddings in {fetch_elapsed:.3f}s")

    # Calculate similarities in Python
    print("\nCalculating similarities in Python...")
    sim_start = time.time()
    similarities = []
    for row in result:
        note_id, summary, embedding = row
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((similarity, note_id, summary))
    sim_elapsed = time.time() - sim_start
    print(f"✓ Calculated {len(similarities):,} similarities in {sim_elapsed:.3f}s")
    print(f"  Rate: {len(similarities)/sim_elapsed:.0f} comparisons/sec")

    # Sort and get top results
    print("\nSorting results...")
    sort_start = time.time()
    similarities.sort(reverse=True)
    top_results = similarities[:limit]
    sort_elapsed = time.time() - sort_start
    print(f"✓ Sorted in {sort_elapsed:.3f}s")

    # Total time
    total_elapsed = model_elapsed + emb_elapsed + fetch_elapsed + sim_elapsed + sort_elapsed

    print("\n" + "-" * 60)
    print("BREAKDOWN:")
    print(f"  Model loading:      {model_elapsed:>8.3f}s ({model_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Query embedding:    {emb_elapsed:>8.3f}s ({emb_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Fetch embeddings:   {fetch_elapsed:>8.3f}s ({fetch_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Calculate similarity: {sim_elapsed:>8.3f}s ({sim_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Sort results:       {sort_elapsed:>8.3f}s ({sort_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  {'TOTAL:':<20} {total_elapsed:>8.3f}s")
    print("-" * 60)

    print("\nTop 3 results:")
    for i, (score, note_id, summary) in enumerate(top_results[:3], 1):
        print(f"{i}. Score: {score:.4f} - {summary[:100]}...")

    return total_elapsed

def test_database_info(con):
    """Get detailed database information."""
    print("\n" + "=" * 60)
    print("DATABASE INFO")
    print("=" * 60)

    # Table sizes
    print("\nTable row counts:")
    notes_count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    status_count = con.execute("SELECT COUNT(*) FROM note_status_history").fetchone()[0]
    print(f"  notes: {notes_count:,}")
    print(f"  note_status_history: {status_count:,}")

    # Check indexes
    print("\nIndexes:")
    indexes = con.execute("""
        SELECT * FROM duckdb_indexes()
    """).fetchall()
    if indexes:
        for idx in indexes:
            print(f"  {idx}")
    else:
        print("  No indexes found")

    # Memory usage estimate
    embedding_count = con.execute(
        "SELECT COUNT(*) FROM notes WHERE summary_embedding IS NOT NULL"
    ).fetchone()[0]
    # 384 dims * 4 bytes (float32) per embedding
    memory_mb = (embedding_count * 384 * 4) / (1024**2)
    print(f"\nEstimated memory for all embeddings: {memory_mb:.2f} MB")

def test_semantic_search_faiss(con, query_text="election misinformation", limit=10):
    """Test semantic search using FAISS index (optimized approach)."""
    print("\n" + "=" * 60)
    print(f"SEMANTIC SEARCH TEST - FAISS INDEX")
    print("=" * 60)
    print(f"Query: '{query_text}'")
    print(f"Looking for top {limit} results")

    # Check if FAISS index exists
    if not Path(index_path).exists():
        print("\n❌ FAISS index not found!")
        print(f"   Expected at: {index_path}")
        print("   Run rebuild_index.py first to create the index.")
        return None

    # Load model
    print("\nLoading embedding model...")
    model_start = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model_elapsed = time.time() - model_start
    print(f"✓ Model loaded in {model_elapsed:.3f}s")

    # Generate query embedding
    print("\nGenerating query embedding...")
    emb_start = time.time()
    query_embedding = model.encode(query_text)
    emb_elapsed = time.time() - emb_start
    print(f"✓ Query embedding generated in {emb_elapsed:.3f}s")

    # Load FAISS index
    print("\nLoading FAISS index...")
    load_start = time.time()
    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        note_ids = pickle.load(f)
    load_elapsed = time.time() - load_start
    print(f"✓ Index loaded in {load_elapsed:.3f}s")
    print(f"  Total vectors: {index.ntotal:,}")

    # Search using FAISS
    print("\nSearching with FAISS...")
    search_start = time.time()
    query_np = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_np)
    similarities, indices = index.search(query_np, limit)
    search_elapsed = time.time() - search_start
    print(f"✓ Search completed in {search_elapsed:.3f}s")
    print(f"  Rate: {limit/search_elapsed:.0f} results/sec")

    # Get note IDs
    top_note_ids = [note_ids[idx] for idx in indices[0]]

    # Fetch details from database
    print("\nFetching full note details from database...")
    fetch_start = time.time()
    placeholders = ",".join(["?" for _ in top_note_ids])
    result = con.execute(f"""
        SELECT noteId, summary
        FROM notes
        WHERE noteId IN ({placeholders})
    """, top_note_ids).fetchall()
    fetch_elapsed = time.time() - fetch_start
    print(f"✓ Fetched {len(result)} notes in {fetch_elapsed:.3f}s")

    # Total time
    total_elapsed = model_elapsed + emb_elapsed + load_elapsed + search_elapsed + fetch_elapsed

    print("\n" + "-" * 60)
    print("BREAKDOWN:")
    print(f"  Model loading:      {model_elapsed:>8.3f}s ({model_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Query embedding:    {emb_elapsed:>8.3f}s ({emb_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Load FAISS index:   {load_elapsed:>8.3f}s ({load_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  FAISS search:       {search_elapsed:>8.3f}s ({search_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  Fetch from DB:      {fetch_elapsed:>8.3f}s ({fetch_elapsed/total_elapsed*100:>5.1f}%)")
    print(f"  {'TOTAL:':<20} {total_elapsed:>8.3f}s")
    print("-" * 60)

    print("\nTop 3 results:")
    note_id_to_summary = {row[0]: row[1] for row in result}
    for i, (score, note_id) in enumerate(zip(similarities[0][:3], top_note_ids[:3]), 1):
        summary = note_id_to_summary.get(note_id, "N/A")
        print(f"{i}. Score: {score:.4f} - {summary[:100]}...")

    return total_elapsed

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMMUNITY NOTES MCP - PERFORMANCE TEST")
    print("=" * 60)

    try:
        # Connect to database
        con = test_connection()

        # Run tests
        test_database_info(con)
        embedding_count = test_simple_query(con)
        test_filtered_query(con)
        test_embedding_retrieval(con, limit=1000)

        # FAISS semantic search test (optimized approach)
        faiss_time = test_semantic_search_faiss(con, "election misinformation", limit=10)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total embeddings in database: {embedding_count:,}")

        if faiss_time:
            print(f"FAISS semantic search time: {faiss_time:.3f}s")
            print("\nPERFORMANCE ANALYSIS:")
            if faiss_time < 5:
                print("✓ Excellent! Search is fast enough for production use.")
            elif faiss_time < 10:
                print("✓ Good performance")
            else:
                print("⚠️  Slower than expected - check index configuration")
        else:
            print("\n⚠️  FAISS index not available")
            print("\nTo enable fast semantic search:")
            print("1. Run: python rebuild_index.py")
            print("2. Restart your MCP server")
            print("3. Searches will be 10-30x faster!")

        con.close()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
