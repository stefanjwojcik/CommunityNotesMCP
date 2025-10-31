#!/usr/bin/env python3
"""
Simple test script to validate the MCP server works locally.
Tests all the available tools without needing a full MCP client.
"""

import asyncio
import json
from server import (
    query_notes,
    semantic_search,
    get_note_stats,
    get_note_by_id
)

async def test_all_tools():
    """Test all MCP tools."""
    print("=" * 60)
    print("Testing Community Notes MCP Server")
    print("=" * 60)
    print()

    # Test 1: Get statistics
    print("Test 1: get_note_stats")
    print("-" * 60)
    try:
        result = await get_note_stats({})
        content = result[0].text
        stats = json.loads(content)
        print(f"✓ Stats retrieved successfully")
        print(f"  Total notes: {stats.get('total_notes', 0):,}")
        print(f"  Notes with embeddings: {stats.get('notes_with_embeddings', 0):,}")
        if 'by_status' in stats:
            print(f"  Status breakdown:")
            for status, count in stats['by_status'].items():
                print(f"    - {status}: {count:,}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

    # Test 2: Query notes
    print("Test 2: query_notes (helpful notes)")
    print("-" * 60)
    try:
        result = await query_notes({
            "status": "CURRENTLY_RATED_HELPFUL",
            "limit": 3
        })
        content = result[0].text
        data = json.loads(content)
        print(f"✓ Found {data['count']} notes")
        for i, note in enumerate(data['notes'][:2], 1):
            print(f"\n  Note {i}:")
            print(f"    ID: {note['noteId']}")
            print(f"    Status: {note.get('currentStatus', 'N/A')}")
            summary = note.get('summary', '')[:100]
            print(f"    Summary: {summary}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

    # Test 3: Semantic search
    print("Test 3: semantic_search")
    print("-" * 60)
    try:
        result = await semantic_search({
            "query": "misinformation about elections",
            "limit": 3,
            "min_similarity": 0.3
        })
        content = result[0].text
        data = json.loads(content)
        print(f"✓ Found {data['count']} similar notes")
        print(f"  Used FAISS: {data.get('used_faiss', False)}")
        for i, item in enumerate(data['results'][:2], 1):
            note = item['note']
            similarity = item['similarity']
            print(f"\n  Result {i} (similarity: {similarity:.3f}):")
            print(f"    ID: {note['noteId']}")
            summary = note.get('summary', '')[:100]
            print(f"    Summary: {summary}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

    # Test 4: Get note by ID (use first note from stats if available)
    print("Test 4: get_note_by_id")
    print("-" * 60)
    try:
        # Get any note ID from a query
        query_result = await query_notes({"limit": 1})
        query_data = json.loads(query_result[0].text)

        if query_data['notes']:
            note_id = str(query_data['notes'][0]['noteId'])
            result = await get_note_by_id({"note_id": note_id})
            content = result[0].text
            note = json.loads(content)
            print(f"✓ Retrieved note {note_id}")
            print(f"  Status: {note.get('currentStatus', 'N/A')}")
            print(f"  Classification: {note.get('classification', 'N/A')}")
            summary = note.get('summary', '')[:150]
            print(f"  Summary: {summary}...")
        else:
            print("✗ No notes available to test with")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all_tools())
