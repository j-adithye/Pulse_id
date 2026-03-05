"""
fix_embeddings.py - Recompute average embedding from ind_1 and ind_2 only.
Deletes ind_3 and updates avg for all registered users.

Run once:
    python3 fix_embeddings.py
"""

import os
import sqlite3
import numpy as np
import config


def _deserialize(blob):
    return np.frombuffer(blob, dtype=np.float32).copy()


def _serialize(arr):
    return arr.astype(np.float32).tobytes()


def fix_all():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    users = conn.execute("SELECT username FROM users").fetchall()
    users = [row[0] for row in users]

    if not users:
        print("No users found in database.")
        conn.close()
        return

    print("Found " + str(len(users)) + " user(s): " + ", ".join(users))
    print("")

    for username in users:
        rows = conn.execute(
            "SELECT emb_type, embedding FROM embeddings WHERE username = ?",
            (username,)
        ).fetchall()

        emb_map = {emb_type: _deserialize(blob) for emb_type, blob in rows}

        if "ind_1" not in emb_map or "ind_2" not in emb_map:
            print("[SKIP] " + username + " -- missing ind_1 or ind_2")
            continue

        # Recompute average from ind_1 and ind_2 only
        new_avg = np.mean([emb_map["ind_1"], emb_map["ind_2"]], axis=0)
        new_avg = new_avg / (np.linalg.norm(new_avg) + 1e-10)

        # Update avg embedding
        conn.execute(
            "UPDATE embeddings SET embedding = ? WHERE username = ? AND emb_type = ?",
            (_serialize(new_avg), username, "avg"),
        )

        # Delete ind_3
        conn.execute(
            "DELETE FROM embeddings WHERE username = ? AND emb_type = ?",
            (username, "ind_3"),
        )

        print("[OK] " + username + " -- avg recomputed from ind_1+ind_2, ind_3 deleted")

    conn.commit()
    conn.close()
    print("")
    print("Done.")


if __name__ == "__main__":
    fix_all()