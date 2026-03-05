"""
embeddings.py - SQLite storage for user embeddings

Schema:
    users table      : username, created_at
    embeddings table : username, emb_type (avg|ind_1|ind_2|ind_3), embedding blob
"""

import os
import sqlite3
import numpy as np
from datetime import datetime
import config


def _get_conn():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username   TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
            emb_type  TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
    """)
    conn.commit()


def _serialize(arr):
    return arr.astype(np.float32).tobytes()


def _deserialize(blob):
    return np.frombuffer(blob, dtype=np.float32).copy()


def user_exists(username):
    with _get_conn() as conn:
        _init_db(conn)
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row is not None


def save_embeddings(username, individual_embs, avg_emb):
    """
    Save 3 individual embeddings + 1 average for a user.

    Args:
        username        : str
        individual_embs : list of 3 float32 arrays shape (64,)
        avg_emb         : float32 array shape (64,)
    """
    with _get_conn() as conn:
        _init_db(conn)
        conn.execute(
            "INSERT INTO users (username, created_at) VALUES (?, ?)",
            (username, datetime.now().isoformat()),
        )
        conn.execute(
            "INSERT INTO embeddings (username, emb_type, embedding) VALUES (?, ?, ?)",
            (username, "avg", _serialize(avg_emb)),
        )
        for i, emb in enumerate(individual_embs):
            conn.execute(
                "INSERT INTO embeddings (username, emb_type, embedding) VALUES (?, ?, ?)",
                (username, "ind_" + str(i + 1), _serialize(emb)),
            )
        conn.commit()
    print("[embeddings] Saved 4 embeddings for: " + username)


def load_embeddings(username):
    """
    Load embeddings for a user.
    Returns: (individual_embs list of 3 arrays, avg_emb array)
    """
    with _get_conn() as conn:
        _init_db(conn)
        rows = conn.execute(
            "SELECT emb_type, embedding FROM embeddings WHERE username = ? ORDER BY emb_type",
            (username,),
        ).fetchall()

    avg_emb = None
    individual_embs = []
    for emb_type, blob in rows:
        arr = _deserialize(blob)
        if emb_type == "avg":
            avg_emb = arr
        else:
            individual_embs.append(arr)

    if avg_emb is None:
        raise ValueError("No average embedding found for: " + username)

    return individual_embs, avg_emb


def load_all_embeddings():
    """
    Load average embeddings for all users.
    Used by identify() in site 2 (1:N).
    Returns: dict {username: avg_emb}
    """
    with _get_conn() as conn:
        _init_db(conn)
        rows = conn.execute(
            "SELECT username, embedding FROM embeddings WHERE emb_type = ?",
            ("avg",),
        ).fetchall()
    return {username: _deserialize(blob) for username, blob in rows}


def list_users():
    """Return list of all registered usernames ordered by registration time."""
    with _get_conn() as conn:
        _init_db(conn)
        rows = conn.execute(
            "SELECT username FROM users ORDER BY created_at"
        ).fetchall()
    return [row[0] for row in rows]


def delete_user(username):
    """Delete a user and all their embeddings (cascade)."""
    if not user_exists(username):
        return {"success": False, "message": "User not found: " + username}
    with _get_conn() as conn:
        _init_db(conn)
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
    print("[embeddings] Deleted user: " + username)
    return {"success": True, "message": "User deleted: " + username}