#!/usr/bin/env python3
"""
Unified setup script for Community Notes MCP.

Supports two modes:
- Demo: Creates a small 1000-note database for quick testing
- Prod: Downloads and updates full production database
"""

import duckdb
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import subprocess
import tempfile
import zipfile
import argparse
from typing import Tuple, Optional

# Default URLs for latest Community Notes data
DEFAULT_NOTES_URL = "https://ton.twimg.com/birdwatch-public-data/2025/10/29/notes/notes-00000.zip"
DEFAULT_STATUS_URL = "https://ton.twimg.com/birdwatch-public-data/2025/10/29/noteStatusHistory/noteStatusHistory-00000.zip"

DEMO_DB = "community_notes_demo.db"
PROD_DB = "community_notes.db"
DEMO_SIZE = 1000

def download_and_extract_zip(url: str, extract_dir: str) -> Optional[str]:
    """Download a zip file from URL and extract it."""
    print(f"Downloading {url}...")
    zip_path = str(Path(extract_dir) / "download.zip")

    try:
        result = subprocess.run(
            ["curl", "-L", "-o", zip_path, url],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Downloaded")

        print(f"Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        tsv_files = list(Path(extract_dir).glob("*.tsv"))
        if not tsv_files:
            print(f"❌ No TSV files found in zip")
            return None

        extracted_file = str(tsv_files[0])
        print(f"✓ Extracted {Path(extracted_file).name}")
        return extracted_file

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_database(db_path: str):
    """Create DuckDB database with schema."""
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            noteId BIGINT PRIMARY KEY,
            noteAuthorParticipantId VARCHAR,
            createdAtMillis BIGINT,
            tweetId BIGINT,
            classification VARCHAR,
            believable VARCHAR,
            harmful VARCHAR,
            validationDifficulty VARCHAR,
            misleadingOther BOOLEAN,
            misleadingFactualError BOOLEAN,
            misleadingManipulatedMedia BOOLEAN,
            misleadingOutdatedInformation BOOLEAN,
            misleadingMissingImportantContext BOOLEAN,
            misleadingUnverifiedClaimAsFact BOOLEAN,
            misleadingSatire BOOLEAN,
            notMisleadingOther BOOLEAN,
            notMisleadingFactuallyCorrect BOOLEAN,
            notMisleadingOutdatedButNotWhenWritten BOOLEAN,
            notMisleadingClearlySatire BOOLEAN,
            notMisleadingPersonalOpinion BOOLEAN,
            trustworthySources BOOLEAN,
            summary TEXT,
            isMediaNote BOOLEAN,
            summary_embedding FLOAT[]
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS note_status_history (
            noteId BIGINT,
            noteAuthorParticipantId VARCHAR,
            createdAtMillis BIGINT,
            timestampMillisOfFirstNonNMRStatus BIGINT,
            firstNonNMRStatus VARCHAR,
            timestampMillisOfCurrentStatus BIGINT,
            currentStatus VARCHAR,
            timestampMillisOfLatestNonNMRStatus BIGINT,
            mostRecentNonNMRStatus VARCHAR,
            timestampMillisOfStatusLock BIGINT,
            lockedStatus VARCHAR,
            timestampMillisOfRetroLock BIGINT,
            currentCoreStatus VARCHAR,
            currentExpansionStatus VARCHAR,
            currentGroupStatus VARCHAR,
            currentDecidedBy VARCHAR,
            currentModelingGroup VARCHAR,
            timestampMillisOfMostRecentStatusChange BIGINT,
            timestampMillisOfNmrDueToMinStableCrhTime BIGINT,
            currentMultiGroupStatus VARCHAR,
            currentModelingMultiGroup VARCHAR,
            timestampMinuteOfFinalScoringOutput BIGINT,
            timestampMillisOfFirstNmrDueToMinStableCrhTime BIGINT,
            PRIMARY KEY (noteId)
        )
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_current_status
        ON note_status_history(currentStatus)
    """)

    return con

def check_demo_database(db_path: str) -> bool:
    """Check if demo database is valid (has 1000 notes with embeddings)."""
    if not Path(db_path).exists():
        return False

    try:
        con = duckdb.connect(db_path, read_only=True)
        count = con.execute(
            "SELECT COUNT(*) FROM notes WHERE summary_embedding IS NOT NULL"
        ).fetchone()[0]
        con.close()

        if count >= DEMO_SIZE:
            print(f"✓ Valid demo database found with {count:,} notes")
            return True
        else:
            print(f"⚠️  Demo database has only {count:,} notes (expected {DEMO_SIZE:,})")
            return False
    except Exception as e:
        print(f"⚠️  Demo database is corrupted: {e}")
        return False

def load_and_embed_notes(
    notes_path: str,
    sample_size: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> pd.DataFrame:
    """Load notes, optionally sample, and generate embeddings."""
    print(f"\nLoading notes from {Path(notes_path).name}...")
    notes_df = pd.read_csv(notes_path, sep='\t', low_memory=False)
    print(f"✓ Loaded {len(notes_df):,} notes")

    # Filter to notes with summaries
    notes_df = notes_df[notes_df['summary'].notna() & (notes_df['summary'] != '')].copy()
    print(f"✓ {len(notes_df):,} notes have summaries")

    # Sample if requested
    if sample_size and len(notes_df) > sample_size:
        print(f"Sampling {sample_size:,} random notes...")
        notes_df = notes_df.sample(n=sample_size, random_state=42)
        print(f"✓ Sampled {len(notes_df):,} notes")

    # Load embedding model
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded")

    # Generate embeddings
    print("Generating embeddings...")
    summaries = notes_df['summary'].tolist()
    embeddings = model.encode(summaries, show_progress_bar=True, batch_size=32)

    notes_df['summary_embedding'] = [emb.tolist() for emb in embeddings]
    print(f"✓ Generated {len(embeddings):,} embeddings")

    return notes_df

def load_status_history(
    status_path: str,
    note_ids: set
) -> pd.DataFrame:
    """Load status history and filter to matching note IDs."""
    print(f"\nLoading status history from {Path(status_path).name}...")
    status_df = pd.read_csv(status_path, sep='\t', low_memory=False)
    print(f"✓ Loaded {len(status_df):,} status records")

    # Filter to only note IDs we have
    status_df = status_df[status_df['noteId'].isin(note_ids)].copy()
    print(f"✓ Filtered to {len(status_df):,} matching status records")

    return status_df

def build_faiss_index(db_path: str, index_suffix: str = ""):
    """Build FAISS index from database embeddings."""
    script_dir = Path(__file__).parent.absolute()
    index_path = str(script_dir / f"notes_index{index_suffix}.faiss")
    mapping_path = str(script_dir / f"note_ids_mapping{index_suffix}.pkl")

    print(f"\n{'='*60}")
    print("BUILDING FAISS INDEX")
    print(f"{'='*60}\n")

    con = duckdb.connect(db_path, read_only=True)

    print("Loading embeddings from database...")
    rows = con.execute("""
        SELECT noteId, summary_embedding
        FROM notes
        WHERE summary_embedding IS NOT NULL
    """).fetchall()

    if len(rows) == 0:
        print("❌ No embeddings found!")
        con.close()
        return False

    print(f"✓ Loaded {len(rows):,} embeddings")

    note_ids = [row[0] for row in rows]
    embeddings = np.array([row[1] for row in rows], dtype=np.float32)

    print(f"Building FAISS index (dimension={embeddings.shape[1]})...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"✓ Index built with {index.ntotal:,} vectors")

    print(f"Saving index to {Path(index_path).name}...")
    faiss.write_index(index, index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump(note_ids, f)

    print(f"✓ Index saved")
    con.close()
    return True

def setup_demo():
    """Setup demo database with 1000 notes."""
    script_dir = Path(__file__).parent.absolute()
    db_path = str(script_dir / DEMO_DB)

    print(f"\n{'='*60}")
    print(f"DEMO MODE SETUP")
    print(f"{'='*60}\n")

    # Check if valid demo exists
    if check_demo_database(db_path):
        print(f"\n✓ Demo database is ready: {DEMO_DB}")
        print(f"✓ Contains {DEMO_SIZE:,} notes with embeddings")

        # Check if FAISS index exists
        index_path = script_dir / "notes_index_demo.faiss"
        if index_path.exists():
            print(f"✓ FAISS index exists")
            print("\n🎉 Demo setup complete! Ready to use.")
            return
        else:
            print("⚠️  FAISS index missing, rebuilding...")
            build_faiss_index(db_path, "_demo")
            print("\n🎉 Demo setup complete! Ready to use.")
            return

    print(f"Setting up fresh demo database...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download data
        print(f"\n{'='*60}")
        print("DOWNLOADING DATA")
        print(f"{'='*60}\n")

        notes_path = download_and_extract_zip(DEFAULT_NOTES_URL, temp_dir)
        if not notes_path:
            print("❌ Failed to download notes")
            sys.exit(1)

        status_path = download_and_extract_zip(DEFAULT_STATUS_URL, temp_dir)
        if not status_path:
            print("❌ Failed to download status history")
            sys.exit(1)

        # Load and sample data
        print(f"\n{'='*60}")
        print(f"CREATING DEMO DATABASE ({DEMO_SIZE:,} notes)")
        print(f"{'='*60}")

        notes_df = load_and_embed_notes(notes_path, sample_size=DEMO_SIZE)
        note_ids = set(notes_df['noteId'].tolist())
        status_df = load_status_history(status_path, note_ids)

        # Create database
        print(f"\nCreating database: {DEMO_DB}...")
        if Path(db_path).exists():
            Path(db_path).unlink()

        con = create_database(db_path)

        print("Inserting notes...")
        con.register('notes_temp', notes_df)
        con.execute("INSERT INTO notes SELECT * FROM notes_temp")

        print("Inserting status history...")
        con.register('status_temp', status_df)
        con.execute("INSERT INTO note_status_history SELECT * FROM status_temp")

        note_count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        status_count = con.execute("SELECT COUNT(*) FROM note_status_history").fetchone()[0]

        con.close()

        print(f"\n✓ Database created:")
        print(f"  Notes: {note_count:,}")
        print(f"  Status records: {status_count:,}")

        # Build FAISS index
        build_faiss_index(db_path, "_demo")

    print(f"\n{'='*60}")
    print("🎉 DEMO SETUP COMPLETE!")
    print(f"{'='*60}")
    print(f"Database: {DEMO_DB}")
    print(f"Notes: {DEMO_SIZE:,}")
    print(f"\nNext steps:")
    print(f"1. Update your MCP config to use: {DEMO_DB}")
    print(f"2. Start using the MCP server!")

def setup_prod(notes_url: str, status_url: str):
    """Setup or update production database."""
    script_dir = Path(__file__).parent.absolute()
    db_path = str(script_dir / PROD_DB)

    print(f"\n{'='*60}")
    print(f"PRODUCTION MODE SETUP")
    print(f"{'='*60}\n")

    # Check for existing database
    existing_db = Path(db_path).exists()
    if existing_db:
        print(f"✓ Existing production database found")
        con = duckdb.connect(db_path, read_only=True)
        existing_count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        con.close()
        print(f"  Current notes: {existing_count:,}")
    else:
        print(f"Creating new production database")
        existing_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download data
        print(f"\n{'='*60}")
        print("DOWNLOADING DATA")
        print(f"{'='*60}\n")

        notes_path = download_and_extract_zip(notes_url, temp_dir)
        if not notes_path:
            print("❌ Failed to download notes")
            sys.exit(1)

        status_path = download_and_extract_zip(status_url, temp_dir)
        if not status_path:
            print("❌ Failed to download status history")
            sys.exit(1)

        if existing_db:
            # Update mode
            print(f"\n{'='*60}")
            print("UPDATING EXISTING DATABASE")
            print(f"{'='*60}")

            from update_data import (
                get_existing_note_ids,
                get_existing_status_ids,
                load_and_filter_new_notes,
                load_and_filter_new_status,
                update_database
            )

            existing_note_ids = get_existing_note_ids(db_path)
            existing_status_ids = get_existing_status_ids(db_path)

            new_notes_df, updated_notes_df = load_and_filter_new_notes(
                notes_path, existing_note_ids
            )
            new_status_df, updated_status_df = load_and_filter_new_status(
                status_path, existing_status_ids
            )

            if (len(new_notes_df) == 0 and len(updated_notes_df) == 0 and
                len(new_status_df) == 0 and len(updated_status_df) == 0):
                print("\n✓ Database is up to date!")
            else:
                update_database(new_notes_df, updated_notes_df,
                              new_status_df, updated_status_df, db_path)

        else:
            # Fresh install mode
            print(f"\n{'='*60}")
            print("CREATING NEW DATABASE")
            print(f"{'='*60}")

            notes_df = load_and_embed_notes(notes_path)
            note_ids = set(notes_df['noteId'].tolist())
            status_df = load_status_history(status_path, note_ids)

            con = create_database(db_path)

            print("\nInserting notes...")
            con.register('notes_temp', notes_df)
            con.execute("INSERT INTO notes SELECT * FROM notes_temp")

            print("Inserting status history...")
            con.register('status_temp', status_df)
            con.execute("INSERT INTO note_status_history SELECT * FROM status_temp")

            note_count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            status_count = con.execute("SELECT COUNT(*) FROM note_status_history").fetchone()[0]

            con.close()

            print(f"\n✓ Database created:")
            print(f"  Notes: {note_count:,}")
            print(f"  Status records: {status_count:,}")

        # Build FAISS index
        build_faiss_index(db_path)

    print(f"\n{'='*60}")
    print("🎉 PRODUCTION SETUP COMPLETE!")
    print(f"{'='*60}")
    print(f"Database: {PROD_DB}")

def main():
    parser = argparse.ArgumentParser(
        description="Setup Community Notes MCP (Demo or Production)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo setup (1000 notes, ~30 seconds)
  python setup.py --demo

  # Full production setup (all notes)
  python setup.py --prod

  # Production with custom URLs
  python setup.py --prod --notes-url <url> --status-url <url>
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Setup demo mode (1000 notes, fast)"
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Setup production mode (full database)"
    )
    parser.add_argument(
        "--notes-url",
        type=str,
        default=DEFAULT_NOTES_URL,
        help="URL for notes data (prod mode only)"
    )
    parser.add_argument(
        "--status-url",
        type=str,
        default=DEFAULT_STATUS_URL,
        help="URL for status history (prod mode only)"
    )

    args = parser.parse_args()

    if not args.demo and not args.prod:
        parser.print_help()
        print("\n❌ Error: Must specify either --demo or --prod")
        sys.exit(1)

    if args.demo and args.prod:
        print("❌ Error: Cannot specify both --demo and --prod")
        sys.exit(1)

    try:
        if args.demo:
            setup_demo()
        else:
            setup_prod(args.notes_url, args.status_url)

    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
