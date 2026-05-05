from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "users.db"
PBKDF2_ITERATIONS = 100_000
SALT_SIZE = 16


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def init_db() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Users (
                user_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                creation_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Sessions (
                session_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                user_ID INTEGER NOT NULL,
                Text_type TEXT,
                Language TEXT,
                status TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                expire_time DATETIME,
                FOREIGN KEY (user_ID) REFERENCES Users (user_ID) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Storage_paths (
                storage_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                user_ID INTEGER NOT NULL,
                session_ID INTEGER NOT NULL,
                image_path TEXT,
                audio_path TEXT,
                slides_path TEXT,
                Textbook_path TEXT,
                slides_output_path TEXT,
                video_output_path TEXT,
                FOREIGN KEY (user_ID) REFERENCES Users (user_ID) ON DELETE CASCADE,
                FOREIGN KEY (session_ID) REFERENCES Sessions (session_ID) ON DELETE CASCADE
            )
            """
        )

        conn.commit()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_SIZE)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"{PBKDF2_ITERATIONS}${salt.hex()}${derived_key.hex()}"


def verify_password(password: str, stored_password: str) -> bool:
    try:
        iterations_str, salt_hex, hash_hex = stored_password.split("$", 2)
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False

    candidate_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate_hash, expected_hash)


def get_user_by_email(email: str) -> sqlite3.Row | None:
    normalized_email = normalize_email(email)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_ID, Email, password, creation_date FROM Users WHERE Email = ?",
            (normalized_email,),
        )
        return cursor.fetchone()


def create_user(email: str, password: str) -> bool:
    normalized_email = normalize_email(email)
    password_hash = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Users (Email, password) VALUES (?, ?)",
                (normalized_email, password_hash),
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(email: str, password: str) -> tuple[bool, str, sqlite3.Row | None]:
    user = get_user_by_email(email)

    if user is None:
        return False, "No account exists for this email.", None

    if not verify_password(password, user["password"]):
        return False, "Incorrect password.", None

    return True, "Login successful.", user


def get_max_session_id() -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(session_ID), 0) FROM Sessions")
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0


def create_session_record(
    *,
    session_id: int,
    user_id: int,
    text_type: str,
    language: str,
    status: str = "pending",
    expire_time: str | None = None,
) -> int:
    if expire_time is None:
        expire_time = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO Sessions (session_ID, user_ID, Text_type, Language, status, expire_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, text_type, language, status, expire_time),
        )
        conn.commit()
        return session_id


def update_session_status(session_id: int, status: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE Sessions SET status = ? WHERE session_ID = ?",
            (status, session_id),
        )
        conn.commit()


def get_session_record(session_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT session_ID, user_ID, Text_type, Language, status, start_time, expire_time
            FROM Sessions
            WHERE session_ID = ?
            """,
            (session_id,),
        )
        return cursor.fetchone()


def delete_session_record(session_id: int) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Sessions WHERE session_ID = ?", (session_id,))
        conn.commit()


def upsert_storage_paths(
    *,
    user_id: int,
    session_id: int,
    image_path: str | None = None,
    audio_path: str | None = None,
    slides_path: str | None = None,
    textbook_path: str | None = None,
    slides_output_path: str | None = None,
    video_output_path: str | None = None,
) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT storage_ID FROM Storage_paths WHERE session_ID = ?",
            (session_id,),
        )
        existing = cursor.fetchone()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO Storage_paths (
                    user_ID,
                    session_ID,
                    image_path,
                    audio_path,
                    slides_path,
                    Textbook_path,
                    slides_output_path,
                    video_output_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    image_path,
                    audio_path,
                    slides_path,
                    textbook_path,
                    slides_output_path,
                    video_output_path,
                ),
            )
        else:
            cursor.execute(
                """
                UPDATE Storage_paths
                SET user_ID = ?,
                    image_path = ?,
                    audio_path = ?,
                    slides_path = ?,
                    Textbook_path = ?,
                    slides_output_path = ?,
                    video_output_path = ?
                WHERE session_ID = ?
                """,
                (
                    user_id,
                    image_path,
                    audio_path,
                    slides_path,
                    textbook_path,
                    slides_output_path,
                    video_output_path,
                    session_id,
                ),
            )

        conn.commit()


def get_storage_paths(session_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT storage_ID, user_ID, session_ID, image_path, audio_path,
                   slides_path, Textbook_path, slides_output_path, video_output_path
            FROM Storage_paths
            WHERE session_ID = ?
            """,
            (session_id,),
        )
        return cursor.fetchone()


if __name__ == "__main__":
    init_db()
