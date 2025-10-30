#!/usr/bin/env python3
"""
Ingestion script for Community Notes data.
Reads TSV files, generates embeddings, and stores in DuckDB.
"""

import duckdb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
from typing import Optional

def create_database(db_path: str = "community_notes.db"):
    """Create DuckDB database with appropriate schema."""
    con = duckdb.connect(db_path)

    # Create notes table
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

    # Create status history table
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

    # Create index on status for faster filtering
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_current_status
        ON note_status_history(currentStatus)
    """)

    return con

def load_and_embed_notes(
    notes_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 1000
):
    """Load notes and generate embeddings for summaries."""
    print(f"Loading notes from {notes_path}...")

    # Read notes TSV
    notes_df = pd.read_csv(notes_path, sep='\t', low_memory=False)
    print(f"Loaded {len(notes_df)} notes")

    # Filter out notes without summaries
    notes_df = notes_df[notes_df['summary'].notna() & (notes_df['summary'] != '')]
    print(f"Filtered to {len(notes_df)} notes with summaries")

    # Load embedding model
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Generate embeddings in batches
    print("Generating embeddings...")
    summaries = notes_df['summary'].tolist()
    embeddings = []

    for i in range(0, len(summaries), batch_size):
        batch = summaries[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(summaries))}/{len(summaries)} summaries")

    # Add embeddings to dataframe
    notes_df['summary_embedding'] = [emb.tolist() for emb in embeddings]

    return notes_df

def load_status_history(status_path: str):
    """Load note status history."""
    print(f"Loading status history from {status_path}...")
    status_df = pd.read_csv(status_path, sep='\t', low_memory=False)
    print(f"Loaded {len(status_df)} status records")
    return status_df

def ingest_data(
    notes_path: str,
    status_path: str,
    db_path: str = "community_notes.db",
    model_name: str = "all-MiniLM-L6-v2"
):
    """Main ingestion pipeline."""
    # Create database
    print("Creating database...")
    con = create_database(db_path)

    # Load and embed notes
    notes_df = load_and_embed_notes(notes_path, model_name)

    # Load status history
    status_df = load_status_history(status_path)

    # Insert notes
    print("Inserting notes into database...")
    con.execute("DELETE FROM notes")
    con.register('notes_temp', notes_df)
    con.execute("INSERT INTO notes SELECT * FROM notes_temp")

    # Insert status history
    print("Inserting status history into database...")
    con.execute("DELETE FROM note_status_history")
    con.register('status_temp', status_df)
    con.execute("INSERT INTO note_status_history SELECT * FROM status_temp")

    # Verify
    note_count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    status_count = con.execute("SELECT COUNT(*) FROM note_status_history").fetchone()[0]

    print(f"\nIngestion complete!")
    print(f"  Notes: {note_count}")
    print(f"  Status records: {status_count}")

    con.close()

if __name__ == "__main__":
    notes_path = "data/notes-00000.tsv"
    status_path = "data/noteStatusHistory-00000.tsv"

    if not Path(notes_path).exists():
        print(f"Error: {notes_path} not found")
        sys.exit(1)

    if not Path(status_path).exists():
        print(f"Error: {status_path} not found")
        sys.exit(1)

    ingest_data(notes_path, status_path)
