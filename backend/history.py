import ipaddress
import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "observations.db"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                ip TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                country TEXT,
                region TEXT,
                city TEXT,
                organization TEXT,
                asn TEXT,
                abuse_confidence INTEGER,
                UNIQUE(collection_id, ip)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_time
            ON observations(observed_at)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_ip
            ON observations(ip)
        """)


def valid_coordinate(value, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def normalize_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 100 else None


def record_collection(items):
    collection_id = str(uuid4())
    observed_at = utc_now()
    rows = []

    for item in items:
        try:
            ip = str(ipaddress.ip_address(item.get("ip", "")))
        except ValueError:
            continue

        organization = str(item.get("org") or "Unknown")
        match = re.search(r"\bAS(\d+)\b", organization)
        asn = f"AS{match.group(1)}" if match else None

        latitude = valid_coordinate(item.get("latitude"), -90, 90)
        longitude = valid_coordinate(item.get("longitude"), -180, 180)

        if latitude is None or longitude is None:
            latitude = longitude = None

        rows.append((
            collection_id,
            observed_at,
            "abuseipdb_blacklist",
            ip,
            latitude,
            longitude,
            item.get("country"),
            item.get("region"),
            item.get("city"),
            organization,
            asn,
            normalize_score(item.get("abuseConfidence")),
        ))

    with connect() as connection:
        connection.executemany("""
            INSERT OR IGNORE INTO observations (
                collection_id, observed_at, source, ip,
                latitude, longitude, country, region, city,
                organization, asn, abuse_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    return {
        "collection_id": collection_id,
        "observed_at": observed_at,
    }


def time_window(days):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def get_observations(start, end, limit=1000):
    with connect() as connection:
        rows = connection.execute("""
            SELECT *
            FROM observations
            WHERE observed_at >= ? AND observed_at < ?
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
        """, (start, end, limit)).fetchall()

    return [dict(row) for row in rows]


def get_statistics(start, end):
    with connect() as connection:
        row = connection.execute("""
            SELECT
                COUNT(*) AS observation_count,
                COUNT(DISTINCT ip) AS unique_ips,
                COUNT(DISTINCT collection_id) AS collection_count,
                COUNT(DISTINCT CASE
                    WHEN country IS NOT NULL
                        AND country NOT IN ('', 'Unknown')
                    THEN country
                END) AS country_count
            FROM observations
            WHERE observed_at >= ? AND observed_at < ?
        """, (start, end)).fetchone()

    return dict(row)
