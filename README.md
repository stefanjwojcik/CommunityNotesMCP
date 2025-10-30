# TikTok Community Notes MCP Server

A Model Context Protocol (MCP) server for querying and semantically searching TikTok Community Notes. This server provides fast access to community notes data with support for filtering by status (helpful, not helpful, needs more ratings) and semantic similarity search using embeddings.

## Features

- **Fast Query Performance**: Uses DuckDB for efficient structured queries
- **Semantic Search**: Embeddings-based similarity search for finding related notes
- **Comprehensive Filtering**: Filter by status, classification, tweet ID, and more
- **Statistics**: Get insights into the community notes dataset

## Tools Available

### 1. `query_notes`
Query community notes by various criteria.

Parameters:
- `status` (optional): Filter by CURRENTLY_RATED_HELPFUL, CURRENTLY_RATED_NOT_HELPFUL, or NEEDS_MORE_RATINGS
- `classification` (optional): Filter by classification type
- `tweet_id` (optional): Get notes for a specific tweet
- `limit` (optional): Maximum results to return (default: 10)

### 2. `semantic_search`
Find notes with similar meaning to a query phrase using embeddings.

Parameters:
- `query` (required): The search phrase
- `limit` (optional): Maximum results to return (default: 10)
- `status` (optional): Filter by note status
- `min_similarity` (optional): Minimum similarity score 0-1 (default: 0.0)

### 3. `get_note_stats`
Get statistics about the notes database.

Returns counts by status, classification, and other metrics.

### 4. `get_note_by_id`
Retrieve a specific note by its ID.

Parameters:
- `note_id` (required): The note ID to retrieve

## Setup

### Quick Install (Recommended)

Run the automated install script:

**Linux/Mac:**
```bash
./install.sh
```

**Windows:**
```bash
install.bat
```

The script will:
1. Check for required data files
2. Create a virtual environment
3. Install all dependencies
4. Run the data ingestion (may take 15-30 minutes)
5. Provide configuration instructions

### Manual Install

If you prefer to install manually:

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Ingest Data

Run the ingestion script to process the TSV files and generate embeddings:

```bash
python ingest_data.py
```

This will:
- Create a DuckDB database (`community_notes.db`)
- Load notes from `data/notes-00000.tsv`
- Load status history from `data/noteStatusHistory-00000.tsv`
- Generate embeddings for all note summaries using the `all-MiniLM-L6-v2` model

**Note**: The ingestion process may take some time depending on the dataset size (2M+ records). Embeddings are generated in batches of 1000.

### Configure MCP Client

Add the server to your MCP client configuration. For Claude Desktop, edit your config file:

**MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "community-notes": {
      "command": "python",
      "args": ["/path/to/CommunityNotesMCP/server.py"]
    }
  }
}
```

**Note**: If you used the install script, it will display the exact configuration with the correct path.

Restart Claude Desktop or your MCP client to load the new server.

## Usage Examples

### Query Notes by Status

```
Use the query_notes tool to find 5 helpful community notes
```

### Semantic Search

```
Use semantic_search to find notes about "election misinformation"
```

### Get Statistics

```
Use get_note_stats to show me statistics about the community notes dataset
```

### Get Specific Note

```
Use get_note_by_id to retrieve note 1783179305159200982
```

## Database Schema

### `notes` Table
- `noteId`: Unique note identifier
- `tweetId`: Associated tweet ID
- `summary`: Note text content
- `classification`: Note classification type
- `summary_embedding`: 384-dimensional embedding vector
- Various misleading/not misleading flags
- Metadata fields

### `note_status_history` Table
- `noteId`: Links to notes table
- `currentStatus`: Current rating status
- `currentDecidedBy`: Model that decided the status
- Various timestamp fields for status changes

## Technical Details

- **Database**: DuckDB (embedded, fast analytical queries)
- **Embeddings**: sentence-transformers with `all-MiniLM-L6-v2` model (384 dimensions)
- **Similarity**: Cosine similarity for semantic matching
- **Performance**: Indexed queries, batch processing for embeddings

## Data Sources

This server expects two TSV files in the `data/` directory:
- `notes-00000.tsv`: Community notes with summaries and classifications
- `noteStatusHistory-00000.tsv`: Status ratings (helpful/not helpful/needs more ratings)

## Development

To modify the embedding model, edit `ingest_data.py` and change the `model_name` parameter. Popular alternatives:
- `all-MiniLM-L6-v2` (default, 384 dims, fast)
- `all-mpnet-base-v2` (768 dims, more accurate, slower)
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingual support)

## Troubleshooting

### "Database not found" error
Run `python ingest_data.py` to create the database first.

### Slow semantic search
- Reduce the dataset size or limit query results
- Consider using a smaller embedding model
- Pre-filter by status before semantic search

### Out of memory during ingestion
- Reduce batch_size in `ingest_data.py`
- Process notes in chunks
- Use a smaller embedding model

## License

This project is for working with TikTok Community Notes data. Please ensure compliance with TikTok's data usage policies.
