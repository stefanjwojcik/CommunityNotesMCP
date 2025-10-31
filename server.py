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

# Auto-detect which database to use (can be overridden by environment variable)
def detect_database():
    """Detect which database file to use."""
    # First check environment variable
    if "COMMUNITY_NOTES_DB" in os.environ:
        return os.environ["COMMUNITY_NOTES_DB"]

    # Check for full database first (preferred)
    full_db = script_dir / "community_notes.db"
    if full_db.exists():
        print(f"Using full database: {full_db.name}")
        return "community_notes.db"

    # Fall back to demo database
    demo_db = script_dir / "community_notes_demo.db"
    if demo_db.exists():
        print(f"Using demo database: {demo_db.name}")
        return "community_notes_demo.db"

    # Default to full database name (will fail later with helpful error)
    return "community_notes.db"

db_name = detect_database()
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
        ),
        Tool(
            name="analyze_status_flips",
            description=(
                "Analyze how often community notes change status (flip) over time. "
                "Shows the count and rate of notes that changed from their initial status to a different status, "
                "with optional time-based trends to see how flipping behavior has evolved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "time_period": {
                        "type": "string",
                        "description": "Group results by time period for trend analysis",
                        "enum": ["day", "week", "month", "year"],
                        "default": "month"
                    },
                    "min_age_days": {
                        "type": "integer",
                        "description": "Only include notes at least this many days old (default: 7)",
                        "default": 7
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include sample notes that flipped for each transition type",
                        "default": False
                    }
                }
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
        elif name == "analyze_status_flips":
            return await analyze_status_flips(arguments)
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

async def analyze_status_flips(args: Dict[str, Any]) -> List[TextContent]:
    """Analyze status flips (notes that changed status over time)."""
    time_period = args.get("time_period", "month")
    min_age_days = args.get("min_age_days", 7)
    include_details = args.get("include_details", False)

    con = get_connection()

    # Calculate cutoff time (min_age_days ago)
    import time
    current_time_millis = int(time.time() * 1000)
    cutoff_time_millis = current_time_millis - (min_age_days * 24 * 60 * 60 * 1000)

    # Time period grouping for SQL
    time_groupings = {
        "day": "DATE_TRUNC('day', TO_TIMESTAMP(createdAtMillis / 1000))",
        "week": "DATE_TRUNC('week', TO_TIMESTAMP(createdAtMillis / 1000))",
        "month": "DATE_TRUNC('month', TO_TIMESTAMP(createdAtMillis / 1000))",
        "year": "DATE_TRUNC('year', TO_TIMESTAMP(createdAtMillis / 1000))"
    }
    time_group_expr = time_groupings.get(time_period, time_groupings["month"])

    # Overall flip statistics
    flip_stats_query = """
        SELECT
            COUNT(*) as total_notes,
            SUM(CASE
                WHEN firstNonNMRStatus IS NOT NULL
                     AND currentStatus IS NOT NULL
                     AND firstNonNMRStatus != currentStatus
                THEN 1 ELSE 0
            END) as flipped_notes,
            SUM(CASE
                WHEN firstNonNMRStatus IS NOT NULL
                     AND currentStatus IS NOT NULL
                     AND firstNonNMRStatus != currentStatus
                THEN 1 ELSE 0
            END) * 100.0 / COUNT(*) as flip_rate
        FROM note_status_history
        WHERE createdAtMillis <= ?
          AND firstNonNMRStatus IS NOT NULL
          AND currentStatus IS NOT NULL
    """

    overall_stats = con.execute(flip_stats_query, [cutoff_time_millis]).fetchone()

    # Flip transitions (what status changed to what)
    transitions_query = """
        SELECT
            firstNonNMRStatus as from_status,
            currentStatus as to_status,
            COUNT(*) as count
        FROM note_status_history
        WHERE createdAtMillis <= ?
          AND firstNonNMRStatus IS NOT NULL
          AND currentStatus IS NOT NULL
          AND firstNonNMRStatus != currentStatus
        GROUP BY firstNonNMRStatus, currentStatus
        ORDER BY count DESC
    """

    transitions = con.execute(transitions_query, [cutoff_time_millis]).fetchall()

    # Time-based trend analysis
    trend_query = f"""
        SELECT
            {time_group_expr} as time_period,
            COUNT(*) as total_notes,
            SUM(CASE
                WHEN firstNonNMRStatus IS NOT NULL
                     AND currentStatus IS NOT NULL
                     AND firstNonNMRStatus != currentStatus
                THEN 1 ELSE 0
            END) as flipped_notes,
            SUM(CASE
                WHEN firstNonNMRStatus IS NOT NULL
                     AND currentStatus IS NOT NULL
                     AND firstNonNMRStatus != currentStatus
                THEN 1 ELSE 0
            END) * 100.0 / COUNT(*) as flip_rate
        FROM note_status_history
        WHERE createdAtMillis <= ?
          AND firstNonNMRStatus IS NOT NULL
          AND currentStatus IS NOT NULL
        GROUP BY {time_group_expr}
        ORDER BY time_period DESC
        LIMIT 24
    """

    trends = con.execute(trend_query, [cutoff_time_millis]).fetchall()

    # Build result structure
    result = {
        "analysis_parameters": {
            "min_age_days": min_age_days,
            "time_period": time_period,
            "cutoff_date": time.strftime('%Y-%m-%d', time.localtime(cutoff_time_millis / 1000))
        },
        "overall_statistics": {
            "total_notes": overall_stats[0],
            "flipped_notes": overall_stats[1],
            "flip_rate_percent": round(overall_stats[2], 2)
        },
        "status_transitions": [
            {
                "from_status": trans[0],
                "to_status": trans[1],
                "count": trans[2]
            }
            for trans in transitions
        ],
        "trends_over_time": [
            {
                "period": str(trend[0]),
                "total_notes": trend[1],
                "flipped_notes": trend[2],
                "flip_rate_percent": round(trend[3], 2)
            }
            for trend in trends
        ]
    }

    # Optionally include sample notes for each transition type
    if include_details:
        result["sample_flipped_notes"] = {}
        for trans in transitions[:5]:  # Top 5 transition types
            from_status, to_status = trans[0], trans[1]
            sample_query = """
                SELECT
                    s.noteId,
                    n.summary,
                    s.firstNonNMRStatus,
                    s.currentStatus,
                    s.timestampMillisOfFirstNonNMRStatus,
                    s.timestampMillisOfCurrentStatus,
                    n.tweetId
                FROM note_status_history s
                JOIN notes n ON s.noteId = n.noteId
                WHERE s.createdAtMillis <= ?
                  AND s.firstNonNMRStatus = ?
                  AND s.currentStatus = ?
                LIMIT 3
            """
            samples = con.execute(sample_query, [cutoff_time_millis, from_status, to_status]).fetchall()

            transition_key = f"{from_status} -> {to_status}"
            result["sample_flipped_notes"][transition_key] = [
                {
                    "noteId": sample[0],
                    "summary": sample[1][:200] + "..." if sample[1] and len(sample[1]) > 200 else sample[1],
                    "from_status": sample[2],
                    "to_status": sample[3],
                    "first_status_timestamp": sample[4],
                    "current_status_timestamp": sample[5],
                    "tweetId": sample[6]
                }
                for sample in samples
            ]

    con.close()

    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
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
