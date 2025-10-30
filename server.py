#!/usr/bin/env python3
"""
MCP Server for Community Notes.
Provides tools for querying and semantic search of community notes.
"""

import duckdb
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import os

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Global variables for model, database, and FAISS index
model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.Index] = None
note_id_mapping: Optional[List[str]] = None

# Use absolute path to database file (in same directory as this script)
script_dir = Path(__file__).parent.absolute()

# Default to demo database (can be overridden by environment variable)
import os
db_name = os.environ.get("COMMUNITY_NOTES_DB", "community_notes_demo.db")
db_path = str(script_dir / db_name)

# Determine index suffix based on database
index_suffix = "_demo" if "demo" in db_name else ""
index_path = str(script_dir / f"notes_index{index_suffix}.faiss")
mapping_path = str(script_dir / f"note_ids_mapping{index_suffix}.pkl")

def get_model():
    """Lazy load the embedding model."""
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_faiss_index():
    """Load FAISS index and note ID mapping from disk."""
    global faiss_index, note_id_mapping

    if faiss_index is None:
        if not Path(index_path).exists():
            print(f"WARNING: FAISS index not found at {index_path}")
            print(f"Database: {db_name}")
            print("Run setup.py to create the index for faster searches.")
            print("Falling back to slower full-scan search.")
            return False

        print(f"Loading FAISS index from {Path(index_path).name}...")
        print(f"Database: {db_name}")
        faiss_index = faiss.read_index(index_path)

        with open(mapping_path, "rb") as f:
            note_id_mapping = pickle.load(f)

        print(f"✓ FAISS index loaded: {faiss_index.ntotal:,} vectors")
        return True

    return True

def get_connection():
    """Get database connection."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Please run ingest_data.py first."
        )
    return duckdb.connect(db_path, read_only=True)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def format_note(note: tuple, columns: List[str]) -> Dict[str, Any]:
    """Format a note tuple into a dictionary."""
    note_dict = dict(zip(columns, note))
    # Remove embedding from output
    if 'summary_embedding' in note_dict:
        del note_dict['summary_embedding']
    return note_dict

# Create MCP server
app = Server("community-notes-mcp")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="query_notes",
            description=(
                "Query community notes by various criteria. "
                "Returns structured note data with status information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": (
                            "Filter by note status: CURRENTLY_RATED_HELPFUL, "
                            "CURRENTLY_RATED_NOT_HELPFUL, or NEEDS_MORE_RATINGS"
                        ),
                        "enum": [
                            "CURRENTLY_RATED_HELPFUL",
                            "CURRENTLY_RATED_NOT_HELPFUL",
                            "NEEDS_MORE_RATINGS"
                        ]
                    },
                    "classification": {
                        "type": "string",
                        "description": "Filter by classification (e.g., MISINFORMED_OR_POTENTIALLY_MISLEADING)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "tweet_id": {
                        "type": "string",
                        "description": "Filter by specific tweet ID"
                    }
                }
            }
        ),
        Tool(
            name="semantic_search",
            description=(
                "Perform semantic search on community note summaries. "
                "Finds notes with similar meaning to the query phrase using embeddings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search phrase to match against note summaries"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "status": {
                        "type": "string",
                        "description": "Optional: Filter by note status",
                        "enum": [
                            "CURRENTLY_RATED_HELPFUL",
                            "CURRENTLY_RATED_NOT_HELPFUL",
                            "NEEDS_MORE_RATINGS"
                        ]
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity score (0-1) for results",
                        "default": 0.0
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_note_stats",
            description=(
                "Get statistics about community notes in the database. "
                "Returns counts by status, classification, and other metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_note_by_id",
            description=(
                "Retrieve a specific community note by its note ID. "
                "Returns full note details including summary and status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "The note ID to retrieve"
                    }
                },
                "required": ["note_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "query_notes":
            return await query_notes(arguments)
        elif name == "semantic_search":
            return await semantic_search(arguments)
        elif name == "get_note_stats":
            return await get_note_stats(arguments)
        elif name == "get_note_by_id":
            return await get_note_by_id(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def query_notes(args: Dict[str, Any]) -> List[TextContent]:
    """Query notes by criteria."""
    con = get_connection()

    # Build query
    query = """
        SELECT
            n.noteId,
            n.tweetId,
            n.classification,
            n.summary,
            n.createdAtMillis,
            s.currentStatus,
            s.currentDecidedBy,
            n.trustworthySources,
            n.misleadingFactualError,
            n.misleadingMissingImportantContext
        FROM notes n
        LEFT JOIN note_status_history s ON n.noteId = s.noteId
        WHERE 1=1
    """
    params = []

    if "status" in args and args["status"]:
        query += " AND s.currentStatus = ?"
        params.append(args["status"])

    if "classification" in args and args["classification"]:
        query += " AND n.classification = ?"
        params.append(args["classification"])

    if "tweet_id" in args and args["tweet_id"]:
        query += " AND n.tweetId = ?"
        params.append(int(args["tweet_id"]))

    limit = args.get("limit", 10)
    query += f" LIMIT {limit}"

    # Execute query
    result = con.execute(query, params).fetchall()
    columns = [desc[0] for desc in con.execute(query, params).description]

    # Format results
    notes = [format_note(row, columns) for row in result]

    con.close()

    return [TextContent(
        type="text",
        text=json.dumps({
            "count": len(notes),
            "notes": notes
        }, indent=2)
    )]

async def semantic_search(args: Dict[str, Any]) -> List[TextContent]:
    """Perform semantic search on note summaries using FAISS."""
    query_text = args["query"]
    limit = args.get("limit", 10)
    min_similarity = args.get("min_similarity", 0.0)
    status_filter = args.get("status")

    # Generate query embedding
    embedding_model = get_model()
    query_embedding = embedding_model.encode(query_text)

    # Try to use FAISS index for fast search
    use_faiss = load_faiss_index()

    con = get_connection()

    if use_faiss and faiss_index is not None and note_id_mapping is not None:
        # FAST PATH: Use FAISS index
        # Normalize query embedding for cosine similarity
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        # Search for more results than needed (for filtering by status)
        search_k = limit * 10 if status_filter else limit
        similarities_scores, indices = faiss_index.search(query_np, search_k)

        # Get note IDs from indices
        top_note_ids = [note_id_mapping[idx] for idx in indices[0]]

        # Fetch full note details from database
        placeholders = ",".join(["?" for _ in top_note_ids])
        sql_query = f"""
            SELECT
                n.noteId,
                n.tweetId,
                n.classification,
                n.summary,
                n.createdAtMillis,
                s.currentStatus,
                s.currentDecidedBy,
                n.trustworthySources
            FROM notes n
            LEFT JOIN note_status_history s ON n.noteId = s.noteId
            WHERE n.noteId IN ({placeholders})
        """

        if status_filter:
            sql_query += f" AND s.currentStatus = ?"
            result = con.execute(sql_query, top_note_ids + [status_filter]).fetchall()
        else:
            result = con.execute(sql_query, top_note_ids).fetchall()

        columns = [desc[0] for desc in con.description]

        # Build results with similarity scores
        # Create a mapping from noteId to similarity score
        note_id_to_similarity = {
            note_id_mapping[idx]: float(score)
            for idx, score in zip(indices[0], similarities_scores[0])
        }

        similarities = []
        for row in result:
            note_dict = dict(zip(columns, row))
            note_id = note_dict['noteId']
            similarity = note_id_to_similarity.get(note_id, 0.0)

            if similarity >= min_similarity:
                similarities.append({
                    "similarity": similarity,
                    "note": note_dict
                })

        # Sort by similarity and limit
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        similarities = similarities[:limit]

    else:
        # SLOW PATH: Fall back to full scan if FAISS not available
        query_embedding_list = query_embedding.tolist()

        sql_query = """
            SELECT
                n.noteId,
                n.tweetId,
                n.classification,
                n.summary,
                n.summary_embedding,
                n.createdAtMillis,
                s.currentStatus,
                s.currentDecidedBy,
                n.trustworthySources
            FROM notes n
            LEFT JOIN note_status_history s ON n.noteId = s.noteId
            WHERE n.summary_embedding IS NOT NULL
        """

        if status_filter:
            sql_query += f" AND s.currentStatus = '{status_filter}'"

        result = con.execute(sql_query).fetchall()
        columns = [desc[0] for desc in con.execute(sql_query).description]

        # Calculate similarities
        similarities = []
        for row in result:
            note_dict = dict(zip(columns, row))
            embedding = note_dict['summary_embedding']
            similarity = cosine_similarity(query_embedding_list, embedding)

            if similarity >= min_similarity:
                del note_dict['summary_embedding']
                similarities.append({
                    "similarity": float(similarity),
                    "note": note_dict
                })

        # Sort by similarity and limit
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        similarities = similarities[:limit]

    con.close()

    return [TextContent(
        type="text",
        text=json.dumps({
            "query": query_text,
            "count": len(similarities),
            "results": similarities,
            "used_faiss": use_faiss
        }, indent=2)
    )]

async def get_note_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Get statistics about notes in the database."""
    con = get_connection()

    # Get various statistics
    stats = {}

    # Total counts
    stats["total_notes"] = con.execute(
        "SELECT COUNT(*) FROM notes"
    ).fetchone()[0]

    stats["total_status_records"] = con.execute(
        "SELECT COUNT(*) FROM note_status_history"
    ).fetchone()[0]

    # Status breakdown
    status_breakdown = con.execute("""
        SELECT currentStatus, COUNT(*) as count
        FROM note_status_history
        GROUP BY currentStatus
        ORDER BY count DESC
    """).fetchall()
    stats["by_status"] = {status: count for status, count in status_breakdown}

    # Classification breakdown
    classification_breakdown = con.execute("""
        SELECT classification, COUNT(*) as count
        FROM notes
        WHERE classification IS NOT NULL
        GROUP BY classification
        ORDER BY count DESC
    """).fetchall()
    stats["by_classification"] = {
        classification: count
        for classification, count in classification_breakdown
    }

    # Notes with embeddings
    stats["notes_with_embeddings"] = con.execute("""
        SELECT COUNT(*) FROM notes WHERE summary_embedding IS NOT NULL
    """).fetchone()[0]

    con.close()

    return [TextContent(
        type="text",
        text=json.dumps(stats, indent=2)
    )]

async def get_note_by_id(args: Dict[str, Any]) -> List[TextContent]:
    """Get a specific note by ID."""
    note_id = args["note_id"]

    con = get_connection()

    result = con.execute("""
        SELECT
            n.*,
            s.currentStatus,
            s.currentDecidedBy,
            s.timestampMillisOfCurrentStatus
        FROM notes n
        LEFT JOIN note_status_history s ON n.noteId = s.noteId
        WHERE n.noteId = ?
    """, [int(note_id)]).fetchone()

    if not result:
        con.close()
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Note {note_id} not found"})
        )]

    columns = [desc[0] for desc in con.execute("""
        SELECT
            n.*,
            s.currentStatus,
            s.currentDecidedBy,
            s.timestampMillisOfCurrentStatus
        FROM notes n
        LEFT JOIN note_status_history s ON n.noteId = s.noteId
        WHERE n.noteId = ?
    """, [int(note_id)]).description]

    note = format_note(result, columns)

    con.close()

    return [TextContent(
        type="text",
        text=json.dumps(note, indent=2)
    )]

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
