import sqlite3
import os
import re
import json
import struct
import math
import requests
import numpy as np

# Vector database name format: drive_embeddings_{sanitized_email}.db
def get_db_path(user_email):
    if not user_email or user_email == 'default_user':
        raise ValueError("Invalid user email for database access")
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_email)
    os.makedirs('embedded_files', exist_ok=True)
    return os.path.join('embedded_files', f"drive_embeddings_{sanitized}.db")

def init_db(user_email):
    """Initialize SQLite database for user's vectors and documents with self-migration"""
    db_path = get_db_path(user_email)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Initialize documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            file_id TEXT PRIMARY KEY,
            user_email TEXT,
            file_name TEXT,
            file_type TEXT,
            full_text TEXT,
            indexed_at TEXT
        )
    """)
    
    # 2. Initialize chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            file_id TEXT,
            user_email TEXT,
            chunk_text TEXT,
            embedding BLOB,
            chunk_index INTEGER,
            name TEXT,
            path TEXT,
            text TEXT,
            id TEXT
        )
    """)
    
    # Check for missing columns in existing chunks table (self-migration)
    cursor.execute("PRAGMA table_info(chunks)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Add any missing columns
    migrations = {
        "chunk_id": "TEXT PRIMARY KEY",
        "user_email": "TEXT",
        "chunk_text": "TEXT",
        "id": "TEXT"
    }
    for col_name, col_def in migrations.items():
        if col_name not in columns:
            try:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def.replace(' PRIMARY KEY', '')}")
            except Exception as e:
                print(f"Migration error adding {col_name}: {e}")
                
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    conn.commit()
    conn.close()

# Serialization helpers using compact float binary array format
def serialize_vector(vector):
    if not vector:
        return b""
    return struct.pack(f"{len(vector)}f", *vector)

def deserialize_vector(blob):
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))

# API Embeddings Generator
def get_embeddings_batch(texts, api_key, provider='nvidia', is_query=False):
    """
    Generate embeddings for a list of texts using Nvidia REST API.
    Returns a list of float arrays (embeddings) or None if request fails.
    """
    if not texts or not api_key:
        return None

    try:
        # REST Endpoint for Nvidia embed-qa-4
        url = "https://integrate.api.nvidia.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "nvidia/embed-qa-4",
            "input": texts,
            "input_type": "query" if is_query else "passage"
        }
        res = requests.post(url, headers=headers, json=payload, timeout=25)
        if res.status_code == 200:
            res_data = res.json()
            # Sort by index to guarantee ordering matches input texts
            sorted_data = sorted(res_data.get('data', []), key=lambda x: x.get('index', 0))
            embeddings = [item['embedding'] for item in sorted_data]
            return embeddings
        else:
            print(f"Nvidia API Embedding error: Code {res.status_code} - {res.text}")
            return None
    except Exception as e:
        print(f"Exception generating embeddings batch: {e}")
        return None

# Database Operation APIs
from datetime import datetime

def save_document(user_email, file_id, file_name, file_type, full_text):
    """Save full document text to documents table, replacing existing entry"""
    db_path = get_db_path(user_email)
    init_db(user_email) # Safety initialization
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Delete old document entry
        cursor.execute("DELETE FROM documents WHERE file_id = ?", (file_id,))
        
        # Insert new document entry
        indexed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        cursor.execute("""
            INSERT INTO documents (file_id, user_email, file_name, file_type, full_text, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_id, user_email, file_name, file_type, full_text, indexed_at))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving document in DB for file {file_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_document(user_email, file_id):
    """Retrieve document dictionary by file_id from SQLite documents table"""
    db_path = get_db_path(user_email)
    if not os.path.exists(db_path):
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT file_id, file_name, file_type, full_text, indexed_at FROM documents WHERE file_id = ?", (file_id,))
        row = cursor.fetchone()
        if row:
            return {
                "file_id": row[0],
                "file_name": row[1],
                "file_type": row[2],
                "full_text": row[3],
                "indexed_at": row[4]
            }
        return None
    except Exception as e:
        print(f"Error reading document from DB: {e}")
        return None
    finally:
        conn.close()

def save_file_chunks(user_email, file_id, file_name, file_path, chunks, embeddings):
    """Save chunks and their embeddings, replacing existing entries for that file"""
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        return False
        
    db_path = get_db_path(user_email)
    init_db(user_email) # Safety initialization
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Delete old chunks
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        
        # Insert new chunks
        insert_data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{file_id}_{i}"
            serialized = serialize_vector(emb)
            insert_data.append((chunk_id, file_id, user_email, chunk, serialized, i, file_name, file_path, chunk, chunk_id))
            
        cursor.executemany("""
            INSERT INTO chunks (chunk_id, file_id, user_email, chunk_text, embedding, chunk_index, name, path, text, id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving chunks in DB for file {file_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def delete_file_chunks(user_email, file_id):
    """Delete all chunks and documents for a specific file"""
    db_path = get_db_path(user_email)
    if not os.path.exists(db_path):
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        cursor.execute("DELETE FROM documents WHERE file_id = ?", (file_id,))
        conn.commit()
    except Exception as e:
        print(f"Error deleting chunks/document for file {file_id}: {e}")
        conn.rollback()
    finally:
        conn.close()

def rename_file_chunks(user_email, file_id, new_name, new_path):
    """Update name and path for the chunks and document of a renamed file"""
    db_path = get_db_path(user_email)
    if not os.path.exists(db_path):
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE chunks 
            SET name = ?, path = ?
            WHERE file_id = ?
        """, (new_name, new_path, file_id))
        cursor.execute("""
            UPDATE documents 
            SET file_name = ?
            WHERE file_id = ?
        """, (new_name, file_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating renamed file chunks/document: {e}")
        conn.rollback()
    finally:
        conn.close()



def update_folder_paths(user_email, old_path_prefix, new_path_prefix):
    """Update paths for chunks inside a folder that was moved or renamed"""
    db_path = get_db_path(user_email)
    if not os.path.exists(db_path):
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Load all chunks whose paths start with old_path_prefix
        cursor.execute("SELECT id, path FROM chunks WHERE path LIKE ?", (old_path_prefix + "%",))
        rows = cursor.fetchall()
        
        updates = []
        for cid, path in rows:
            if path.startswith(old_path_prefix):
                suffix = path[len(old_path_prefix):]
                new_path = new_path_prefix + suffix
                updates.append((new_path, cid))
                
        if updates:
            cursor.executemany("UPDATE chunks SET path = ? WHERE id = ?", updates)
            conn.commit()
    except Exception as e:
        print(f"Error updating folder paths: {e}")
        conn.rollback()
    finally:
        conn.close()

# Cosine similarity calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

def compute_similarity(v1, v2):
    """Compute cosine similarity between two float vectors"""
    if HAS_NUMPY and np is not None:
        arr1 = np.array(v1)
        arr2 = np.array(v2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / (norm1 * norm2))
    else:
        # Pure Python fallback
        if len(v1) != len(v2):
            return 0.0
        dot = sum(x * y for x, y in zip(v1, v2))
        norm1 = math.sqrt(sum(x * x for x in v1))
        norm2 = math.sqrt(sum(y * y for y in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

def semantic_search(user_email, query_embedding, limit=15, file_id=None, file_ids=None):
    """
    Load all chunks for user (optionally filtered by file_id or a list/set of file_ids) and compute cosine similarity against the query embedding.
    Returns a sorted list of matches: [(chunk_dict, score), ...]
    """
    db_path = get_db_path(user_email)
    if not os.path.exists(db_path) or not query_embedding:
        return []
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        if file_ids:
            placeholders = ",".join(["?"] * len(file_ids))
            cursor.execute(f"SELECT file_id, name, path, chunk_index, text, embedding FROM chunks WHERE file_id IN ({placeholders})", tuple(file_ids))
        elif file_id:
            cursor.execute("SELECT file_id, name, path, chunk_index, text, embedding FROM chunks WHERE file_id = ?", (file_id,))
        else:
            cursor.execute("SELECT file_id, name, path, chunk_index, text, embedding FROM chunks")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"Error loading chunks for semantic search: {e}")
        return []
    finally:
        conn.close()
        
    results = []
    for row in rows:
        fid, name, path, cidx, text, blob = row
        chunk_embedding = deserialize_vector(blob)
        if not chunk_embedding:
            continue
            
        score = compute_similarity(query_embedding, chunk_embedding)
        if score > 0.35: # Cosine similarity threshold filter
            chunk_dict = {
                "file_id": fid,
                "name": name,
                "path": path,
                "chunk_index": cidx,
                "text": text
            }
            results.append((chunk_dict, score))
            
    # Sort by score desc
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]



