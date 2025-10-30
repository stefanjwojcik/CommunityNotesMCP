#!/usr/bin/env python3
"""
Update script for TikTok Community Notes data.
Downloads new data (zip files), diffs with existing data, and updates database incrementally.
"""

import duckdb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import subprocess
import tempfile
import zipfile
from typing import Tuple, Optional
import argparse

# Default URLs for Community Notes data (updated for current format)
DEFAULT_NOTES_URL = "https://ton.twimg.com/birdwatch-public-data/2025/10/29/notes/notes-00000.zip"
DEFAULT_STATUS_URL = "https://ton.twimg.com/birdwatch-public-data/2025/10/29/noteStatusHistory/noteStatusHistory-00000.zip"

def download_and_extract_zip(url: str, extract_dir: str) -> Optional[str]:
    """Download a zip file from URL and extract it."""
    print(f"Downloading {url}...")

    zip_path = str(Path(extract_dir) / "download.zip")

    try:
        # Download with curl
        result = subprocess.run(
            ["curl", "-L", "-o", zip_path, url],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Downloaded to {zip_path}")

        # Extract zip file
        print(f"Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the extracted TSV file
        tsv_files = list(Path(extract_dir).glob("*.tsv"))
        if not tsv_files:
            print(f"❌ No TSV files found in zip")
            return None

        extracted_file = str(tsv_files[0])
        print(f"✓ Extracted to {extracted_file}")
        return extracted_file

    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading: {e}")
        print(f"   stderr: {e.stderr}")
        return None
    except zipfile.BadZipFile as e:
        print(f"❌ Error extracting zip: {e}")
        return None

def get_existing_note_ids(db_path: str) -> set:
    """Get set of existing note IDs from database."""
    if not Path(db_path).exists():
        print("Database doesn't exist - this will be a fresh import")
        return set()

    con = duckdb.connect(db_path, read_only=True)
    result = con.execute("SELECT noteId FROM notes").fetchall()
    con.close()

    existing_ids = {row[0] for row in result}
    print(f"Found {len(existing_ids):,} existing notes in database")
    return existing_ids

def get_existing_status_ids(db_path: str) -> set:
    """Get set of existing note IDs from status history."""
    if not Path(db_path).exists():
        return set()

    con = duckdb.connect(db_path, read_only=True)
    result = con.execute("SELECT noteId FROM note_status_history").fetchall()
    con.close()

    existing_ids = {row[0] for row in result}
    print(f"Found {len(existing_ids):,} existing status records in database")
    return existing_ids

def load_and_filter_new_notes(
    notes_path: str,
    existing_ids: set,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load notes and separate into new vs updated records.
    Generate embeddings only for new notes with summaries.
    """
    print(f"\nLoading notes from {notes_path}...")
    notes_df = pd.read_csv(notes_path, sep='\t', low_memory=False)
    print(f"Loaded {len(notes_df):,} total notes from file")

    # Separate new and existing notes
    notes_df['is_new'] = ~notes_df['noteId'].isin(existing_ids)
    new_notes_df = notes_df[notes_df['is_new']].copy()
    updated_notes_df = notes_df[~notes_df['is_new']].copy()

    print(f"  New notes: {len(new_notes_df):,}")
    print(f"  Potentially updated notes: {len(updated_notes_df):,}")

    # Filter new notes to only those with summaries
    new_notes_with_summary = new_notes_df[
        new_notes_df['summary'].notna() & (new_notes_df['summary'] != '')
    ].copy()

    print(f"  New notes with summaries: {len(new_notes_with_summary):,}")

    if len(new_notes_with_summary) > 0:
        # Load embedding model
        print(f"\nLoading embedding model: {model_name}...")
        model = SentenceTransformer(model_name)

        # Generate embeddings for new notes only
        print("Generating embeddings for new notes...")
        summaries = new_notes_with_summary['summary'].tolist()
        embeddings = []

        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            print(f"  Processed {min(i+batch_size, len(summaries))}/{len(summaries)} summaries")

        # Add embeddings to dataframe
        new_notes_with_summary['summary_embedding'] = [emb.tolist() for emb in embeddings]

        # Remove the is_new column
        new_notes_with_summary = new_notes_with_summary.drop('is_new', axis=1)

    # For updated notes, add NULL embedding column to match schema
    # (we don't regenerate embeddings for existing notes)
    if len(updated_notes_df) > 0:
        updated_notes_df = updated_notes_df.drop('is_new', axis=1)
        # Add summary_embedding column with None values
        updated_notes_df['summary_embedding'] = None

    return new_notes_with_summary, updated_notes_df

def load_and_filter_new_status(
    status_path: str,
    existing_ids: set
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load status history and separate into new vs updated records."""
    print(f"\nLoading status history from {status_path}...")
    status_df = pd.read_csv(status_path, sep='\t', low_memory=False)
    print(f"Loaded {len(status_df):,} total status records from file")

    # Separate new and existing status records
    status_df['is_new'] = ~status_df['noteId'].isin(existing_ids)
    new_status_df = status_df[status_df['is_new']].copy()
    updated_status_df = status_df[~status_df['is_new']].copy()

    print(f"  New status records: {len(new_status_df):,}")
    print(f"  Potentially updated status records: {len(updated_status_df):,}")

    # Remove is_new column
    if len(new_status_df) > 0:
        new_status_df = new_status_df.drop('is_new', axis=1)
    if len(updated_status_df) > 0:
        updated_status_df = updated_status_df.drop('is_new', axis=1)

    return new_status_df, updated_status_df

def update_database(
    new_notes_df: pd.DataFrame,
    updated_notes_df: pd.DataFrame,
    new_status_df: pd.DataFrame,
    updated_status_df: pd.DataFrame,
    db_path: str = "community_notes.db"
):
    """Update database with new and updated records."""
    print(f"\n{'='*60}")
    print("UPDATING DATABASE")
    print(f"{'='*60}")

    con = duckdb.connect(db_path)

    # Insert new notes
    if len(new_notes_df) > 0:
        print(f"\nInserting {len(new_notes_df):,} new notes...")
        con.register('new_notes_temp', new_notes_df)
        con.execute("INSERT INTO notes SELECT * FROM new_notes_temp")
        print("✓ New notes inserted")
    else:
        print("\nNo new notes to insert")

    # Update existing notes (preserve embeddings)
    if len(updated_notes_df) > 0:
        print(f"\nUpdating {len(updated_notes_df):,} existing notes...")
        # Get list of all columns except summary_embedding
        columns_to_update = [col for col in updated_notes_df.columns if col != 'summary_embedding']

        # Build UPDATE statement that preserves embeddings
        # We'll use a more careful approach: update only non-embedding fields
        con.register('updated_notes_temp', updated_notes_df)

        # For each note, update all fields except keep the existing embedding
        con.execute("""
            UPDATE notes
            SET
                noteAuthorParticipantId = updated_notes_temp.noteAuthorParticipantId,
                createdAtMillis = updated_notes_temp.createdAtMillis,
                tweetId = updated_notes_temp.tweetId,
                classification = updated_notes_temp.classification,
                believable = updated_notes_temp.believable,
                harmful = updated_notes_temp.harmful,
                validationDifficulty = updated_notes_temp.validationDifficulty,
                misleadingOther = updated_notes_temp.misleadingOther,
                misleadingFactualError = updated_notes_temp.misleadingFactualError,
                misleadingManipulatedMedia = updated_notes_temp.misleadingManipulatedMedia,
                misleadingOutdatedInformation = updated_notes_temp.misleadingOutdatedInformation,
                misleadingMissingImportantContext = updated_notes_temp.misleadingMissingImportantContext,
                misleadingUnverifiedClaimAsFact = updated_notes_temp.misleadingUnverifiedClaimAsFact,
                misleadingSatire = updated_notes_temp.misleadingSatire,
                notMisleadingOther = updated_notes_temp.notMisleadingOther,
                notMisleadingFactuallyCorrect = updated_notes_temp.notMisleadingFactuallyCorrect,
                notMisleadingOutdatedButNotWhenWritten = updated_notes_temp.notMisleadingOutdatedButNotWhenWritten,
                notMisleadingClearlySatire = updated_notes_temp.notMisleadingClearlySatire,
                notMisleadingPersonalOpinion = updated_notes_temp.notMisleadingPersonalOpinion,
                trustworthySources = updated_notes_temp.trustworthySources,
                summary = updated_notes_temp.summary,
                isMediaNote = updated_notes_temp.isMediaNote
            FROM updated_notes_temp
            WHERE notes.noteId = updated_notes_temp.noteId
        """)
        print("✓ Existing notes updated (embeddings preserved)")
    else:
        print("\nNo notes to update")

    # Insert new status records
    if len(new_status_df) > 0:
        print(f"\nInserting {len(new_status_df):,} new status records...")
        con.register('new_status_temp', new_status_df)
        con.execute("INSERT INTO note_status_history SELECT * FROM new_status_temp")
        print("✓ New status records inserted")
    else:
        print("\nNo new status records to insert")

    # Update existing status records
    if len(updated_status_df) > 0:
        print(f"\nUpdating {len(updated_status_df):,} existing status records...")
        con.register('updated_status_temp', updated_status_df)
        con.execute("INSERT OR REPLACE INTO note_status_history SELECT * FROM updated_status_temp")
        print("✓ Existing status records updated")
    else:
        print("\nNo status records to update")

    # Get final counts
    note_count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    status_count = con.execute("SELECT COUNT(*) FROM note_status_history").fetchone()[0]
    notes_with_embeddings = con.execute(
        "SELECT COUNT(*) FROM notes WHERE summary_embedding IS NOT NULL"
    ).fetchone()[0]

    print(f"\n{'='*60}")
    print("UPDATE COMPLETE")
    print(f"{'='*60}")
    print(f"Total notes in database: {note_count:,}")
    print(f"Total status records: {status_count:,}")
    print(f"Notes with embeddings: {notes_with_embeddings:,}")

    con.close()

    return note_count, status_count

def main():
    """Main update pipeline."""
    parser = argparse.ArgumentParser(
        description="Update Community Notes database with new data",
        epilog="Example: python update_data.py --notes-url https://ton.twimg.com/.../notes-00000.zip --rebuild-index"
    )
    parser.add_argument(
        "--notes-url",
        type=str,
        default=DEFAULT_NOTES_URL,
        help="URL to download notes zip file"
    )
    parser.add_argument(
        "--status-url",
        type=str,
        default=DEFAULT_STATUS_URL,
        help="URL to download status history zip file"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="community_notes.db",
        help="Path to DuckDB database file"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and use existing files in data/ directory"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Automatically rebuild FAISS index after update"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    db_path = str(script_dir / args.db_path)

    # Create temp directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.skip_download:
            print("Skipping download - using existing files")
            notes_path = str(script_dir / "data" / "notes-00000.tsv")
            status_path = str(script_dir / "data" / "noteStatusHistory-00000.tsv")

            if not Path(notes_path).exists() or not Path(status_path).exists():
                print("❌ Error: Files not found in data/ directory")
                sys.exit(1)
        else:
            # Download and extract new data
            print(f"{'='*60}")
            print("DOWNLOADING NEW DATA")
            print(f"{'='*60}\n")

            notes_path = download_and_extract_zip(args.notes_url, temp_dir)
            if not notes_path:
                print("❌ Failed to download/extract notes data")
                sys.exit(1)

            status_path = download_and_extract_zip(args.status_url, temp_dir)
            if not status_path:
                print("❌ Failed to download/extract status history data")
                sys.exit(1)

        # Get existing IDs
        print(f"\n{'='*60}")
        print("CHECKING EXISTING DATA")
        print(f"{'='*60}\n")

        existing_note_ids = get_existing_note_ids(db_path)
        existing_status_ids = get_existing_status_ids(db_path)

        # Load and filter new data
        print(f"\n{'='*60}")
        print("PROCESSING NEW DATA")
        print(f"{'='*60}")

        new_notes_df, updated_notes_df = load_and_filter_new_notes(
            notes_path,
            existing_note_ids
        )

        new_status_df, updated_status_df = load_and_filter_new_status(
            status_path,
            existing_status_ids
        )

        # Update database
        if (len(new_notes_df) == 0 and len(updated_notes_df) == 0 and
            len(new_status_df) == 0 and len(updated_status_df) == 0):
            print("\n✓ No new or updated data found. Database is up to date!")
            return

        update_database(
            new_notes_df,
            updated_notes_df,
            new_status_df,
            updated_status_df,
            db_path
        )

        # Rebuild FAISS index if requested
        if args.rebuild_index:
            print(f"\n{'='*60}")
            print("REBUILDING FAISS INDEX")
            print(f"{'='*60}\n")

            rebuild_script = script_dir / "rebuild_index.py"
            if rebuild_script.exists():
                result = subprocess.run(
                    [sys.executable, str(rebuild_script)],
                    check=False
                )
                if result.returncode == 0:
                    print("\n✓ FAISS index rebuilt successfully")
                else:
                    print("\n⚠️  FAISS index rebuild failed")
                    print("   Run rebuild_index.py manually if needed")
            else:
                print("⚠️  rebuild_index.py not found - skipping index rebuild")
        else:
            print("\n💡 Remember to rebuild the FAISS index:")
            print("   python rebuild_index.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
