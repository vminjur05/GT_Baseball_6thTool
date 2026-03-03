"""
dbmanager.py — GT Baseball 6th Tool Database Manager
======================================================
Free, zero-infrastructure SQLite database for storing game CSV/Parquet data.
Eliminates the need to re-upload files every session.

✅ Update (Feb 2026):
- Blocks duplicate uploads even when the label changes (same CSV content).
- Still blocks duplicate game_label (existing behavior).
- Blocks duplicate file_name (even if label changes).
- Adds friendly, explicit warning messages for duplicates (so it's obvious in UI).

✅ Update (Mar 2026):
- Player records (players table) are now GT-roster-only.
  Opponent pitchers/batters/fielders still appear in pitch data but with
  NULL foreign keys, keeping the player table clean (~35 players, not 262).
- reload_roster() also purges any opponent rows already stored.

How duplicate blocking works:
1) game_label duplicate -> blocked
2) file_name duplicate  -> blocked (if file_name provided)
3) content duplicate    -> blocked using a stable SHA256 of key columns

Schema is designed around the real Derived_Data CSV format:
    Inning, AtBat, PitcherName, BatterName, Result, PitchVelo,
    BatterTimeToFirst, BatterTop, ExitVelo, LaunchAng, ActualDistance,
    BaserunnerMaxSpeed, BaserunnerInitial, BaserunnerSecondary, BaserunnerFinal,
    IsEventPlayer, EventPlayerName, FielderProbability, FielderRouteEfficiency,
    FielderMove, FielderReaction, FielderReactionAngle, FielderTransfer,
    FielderThrow, FielderThrowDistance, FielderMaxSpeed

Usage:
    from dbmanager import GTBaseballDB
    db = GTBaseballDB()                        # creates data/gt_baseball.db
    n, status, msg = db.ingest_csv("csv_data/game1.csv", "GT vs UBC Game 1")
    if status == "warning": print(msg)
    df = db.query_all_games()
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/gt_baseball.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val):
    """Convert numpy/pandas NA types to None for SQLite compatibility."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _bool_to_int(val) -> int:
    """SQLite has no bool; store as 0/1 integer. Never returns None."""
    if val is None:
        return 0
    try:
        if pd.isna(val):
            return 0
    except (TypeError, ValueError):
        pass
    return 1 if bool(val) else 0


def _columns_signature(df: pd.DataFrame) -> str:
    """Stable signature of dataframe structure (ignores order and capitalisation)."""
    cols = sorted([str(c).strip().lower() for c in df.columns])
    return hashlib.sha256("|".join(cols).encode("utf-8")).hexdigest()


def _content_hash(df: pd.DataFrame) -> str:
    """Content fingerprint using key baseball identity columns."""
    candidate_cols = [
        "Inning", "AtBat", "PitcherName", "BatterName", "Result",
        "PitchVelo", "ExitVelo", "LaunchAng",
    ]
    present = [c for c in candidate_cols if c in df.columns]
    if not present:
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
    df_small = df[present].fillna("").astype(str)
    df_small = df_small.sort_values(by=present).reset_index(drop=True)
    return hashlib.sha256(df_small.to_json().encode("utf-8")).hexdigest()


def _categorize_pitch_outcome(result) -> str:
    if result is None:
        return "Unknown"
    result_str = str(result).strip()
    if result_str in ("", "nan", "None", "NaN"):
        return "Unknown"
    r = result_str.lower()
    if any(tok in r for tok in ["ball", "bb", "walk"]) and "in play" not in r:
        return "Ball"
    if any(tok in r for tok in ["called strike", "swinging strike", "strike"]):
        return "Strike"
    if "home run" in r or "homer" in r or r == "hr":
        return "Home Run"
    if "triple" in r:
        return "Triple"
    if "double" in r:
        return "Double"
    if "single" in r:
        return "Single"
    if "ground" in r:
        return "Groundout"
    if "fly" in r:
        return "Flyout"
    if any(tok in r for tok in ["in play", "inplay", "bip", "out"]):
        return "In Play"
    if "foul" in r:
        return "Foul"
    if any(tok in r for tok in ["hit by pitch", "hitbypitch", "hbp"]):
        return "HBP"
    if any(tok in r for tok in ["sb", "stolen base"]):
        return "Stolen Base"
    return "Other"


def _calculate_hit_quality(exit_velo, launch_angle) -> str:
    try:
        ev = float(exit_velo)
        la = float(launch_angle)
    except (TypeError, ValueError):
        return "Unknown"
    if ev >= 95 and 8 <= la <= 32:
        return "Barrel"
    if ev >= 95:
        return "Hard Hit"
    if ev < 60:
        return "Weak Contact"
    return "Medium Contact"


def _clean_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GTBaseballDB:
    """
    SQLite-backed store for GT Baseball 6th Tool data.

    Tables
    ------
    games         — one row per uploaded file / game label
    players       — GT roster players only (opponents stored with NULL FK)
    pitches       — core pitch-by-pitch rows (one per CSV row)
    fielding      — fielding metrics (rows where IsEventPlayer = True)
    baserunning   — baserunning metrics (rows with runner data)
    ingested_files— content-hash ledger to block identical uploads
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("GTBaseballDB ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Create all tables if they don't already exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS games (
            game_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            game_label  TEXT    UNIQUE NOT NULL,
            file_name   TEXT,
            loaded_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS players (
            player_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT    UNIQUE NOT NULL,
            role        TEXT
        );

        CREATE TABLE IF NOT EXISTS pitches (
            pitch_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id           INTEGER NOT NULL REFERENCES games(game_id),
            inning            INTEGER,
            at_bat            INTEGER,
            pitcher_id        INTEGER REFERENCES players(player_id),
            batter_id         INTEGER REFERENCES players(player_id),
            result            TEXT,
            pitch_velo        REAL,
            batter_time_first REAL,
            batter_top        REAL,
            exit_velo         REAL,
            launch_angle      REAL,
            actual_distance   REAL,
            ball_in_play      INTEGER DEFAULT 0,
            pitch_outcome     TEXT,
            hit_quality       TEXT
        );

        CREATE TABLE IF NOT EXISTS fielding (
            fielding_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            pitch_id           INTEGER NOT NULL REFERENCES pitches(pitch_id),
            game_id            INTEGER NOT NULL REFERENCES games(game_id),
            fielder_id         INTEGER REFERENCES players(player_id),
            probability        REAL,
            route_efficiency   REAL,
            move               REAL,
            reaction           REAL,
            reaction_angle     REAL,
            transfer           REAL,
            throw              REAL,
            throw_distance     REAL,
            max_speed          REAL
        );

        CREATE TABLE IF NOT EXISTS baserunning (
            run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            pitch_id      INTEGER NOT NULL REFERENCES pitches(pitch_id),
            game_id       INTEGER NOT NULL REFERENCES games(game_id),
            max_speed     REAL,
            initial_base  REAL,
            secondary     REAL,
            final_base    REAL
        );

        CREATE TABLE IF NOT EXISTS ingested_files (
            file_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            game_label     TEXT NOT NULL,
            file_name      TEXT,
            content_sha256 TEXT NOT NULL UNIQUE,
            columns_sig    TEXT NOT NULL,
            row_count      INTEGER NOT NULL,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_pitches_game    ON pitches(game_id);
        CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher_id);
        CREATE INDEX IF NOT EXISTS idx_pitches_batter  ON pitches(batter_id);
        CREATE INDEX IF NOT EXISTS idx_fielding_game   ON fielding(game_id);
        CREATE INDEX IF NOT EXISTS idx_fielding_player ON fielding(fielder_id);
        CREATE INDEX IF NOT EXISTS idx_baserun_game    ON baserunning(game_id);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_ingested_game_label ON ingested_files(game_label);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ingested_file_name  ON ingested_files(file_name);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ingested_sha        ON ingested_files(content_sha256);
        CREATE INDEX IF NOT EXISTS idx_ingested_files_file_name   ON ingested_files(file_name);
        """

        with self._connect() as conn:
            conn.executescript(ddl)

            # Schema self-heal for older DBs missing columns
            cols = {
                r[1] for r in conn.execute("PRAGMA table_info(ingested_files)").fetchall()
            }
            if "columns_sig" not in cols:
                conn.execute(
                    "ALTER TABLE ingested_files ADD COLUMN columns_sig TEXT NOT NULL DEFAULT ''"
                )
            if "row_count" not in cols:
                conn.execute(
                    "ALTER TABLE ingested_files ADD COLUMN row_count INTEGER NOT NULL DEFAULT 0"
                )

    # ------------------------------------------------------------------
    # GT Roster loader (cached per instance)
    # ------------------------------------------------------------------

    def _load_gt_roster(self) -> set:
        """
        Return a set of lowercase GT player names from scripts/gt_roster.csv.
        Falls back to empty set so nothing breaks if the file is missing.
        """
        if hasattr(self, "_gt_names"):
            return self._gt_names
        roster_path = Path("scripts/gt_roster.csv")
        if not roster_path.exists():
            self._gt_names = set()
            return self._gt_names
        try:
            df = pd.read_csv(roster_path)
            col = next(
                (c for c in ["player_name", "name", "PlayerName", "full_name"]
                 if c in df.columns),
                df.columns[0],
            )
            self._gt_names = set(
                df[col].dropna().astype(str).str.strip().str.lower()
            )
        except Exception:
            self._gt_names = set()
        return self._gt_names

    def _is_gt_player(self, name: str) -> bool:
        """Return True if name appears in the GT roster (case-insensitive)."""
        gt = self._load_gt_roster()
        if not gt:
            return True  # no roster file → treat everyone as GT (safe degradation)
        return str(name).strip().lower() in gt

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _stable_df_hash(self, df: pd.DataFrame) -> str:
        """Compute a stable SHA256 hash for a game's pitch-level content."""
        df = self._normalise_columns(df)
        key_cols = [c for c in [
            "Inning", "AtBat", "PitcherName", "BatterName", "Result",
            "PitchVelo", "ExitVelo", "LaunchAng", "BatterTimeToFirst", "ActualDistance",
        ] if c in df.columns]

        if not key_cols:
            cols = "|".join(sorted([str(c) for c in df.columns]))
            blob = f"{cols}::{len(df)}".encode("utf-8")
            return hashlib.sha256(blob).hexdigest()

        temp = df[key_cols].copy()
        for c in temp.columns:
            if pd.api.types.is_numeric_dtype(temp[c]):
                temp[c] = pd.to_numeric(temp[c], errors="coerce").round(3)
            else:
                temp[c] = temp[c].astype(str).fillna("").map(_clean_str)
        temp = temp.fillna("")
        temp = temp.sort_values(by=key_cols, kind="mergesort").reset_index(drop=True)
        lines = temp.astype(str).agg("|".join, axis=1).tolist()
        blob = ("\n".join(lines)).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _duplicate_reason(
        self,
        df: pd.DataFrame,
        game_label: str,
        file_name: str | None = None,
    ) -> Optional[str]:
        """Returns a reason string if this upload is a duplicate, else None."""
        game_label = _clean_str(game_label)
        file_name = _clean_str(file_name) if file_name else None

        if self.game_exists(game_label):
            return f"⚠️ Game label already exists in the database: '{game_label}'. Not added."

        if file_name:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT game_label, loaded_at FROM games WHERE file_name = ?",
                    (file_name,),
                ).fetchone()
            if row:
                existing_label, loaded_at = row
                return (
                    "⚠️ A file with this exact file name is already stored.\n"
                    f"Existing entry: label='{existing_label}', file_name='{file_name}', added={loaded_at}.\n"
                    "Not added."
                )

        content_hash = self._stable_df_hash(df)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT game_label, file_name, created_at FROM ingested_files WHERE content_sha256 = ?",
                (content_hash,),
            ).fetchone()
        if row:
            prev_label, prev_file, prev_time = row
            return (
                "⚠️ This upload matches a file already in the database (same content).\n"
                f"Existing entry: label='{prev_label}', file='{prev_file}', added={prev_time}.\n"
                "Not added."
            )

        return None

    # ------------------------------------------------------------------
    # Player helpers — GT roster only
    # ------------------------------------------------------------------

    def _get_or_create_player(
        self,
        conn: sqlite3.Connection,
        name: str,
        role: str = None,
    ) -> Optional[int]:
        """
        Return a player_id only for GT roster members.
        Opponent players return None — their pitch rows are still stored
        with NULL pitcher_id / batter_id / fielder_id.
        """
        if not name or str(name).strip() in ("", "nan", "None", "NaN"):
            return None

        name = str(name).strip()

        # GT-only gate — opponents get NULL FK, not a players row
        if not self._is_gt_player(name):
            return None

        conn.execute(
            "INSERT OR IGNORE INTO players (player_name, role) VALUES (?, ?)",
            (name, role),
        )
        row = conn.execute(
            "SELECT player_id, role FROM players WHERE player_name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None

        player_id, existing_role = row
        if role and existing_role and existing_role != role and existing_role != "multiple":
            conn.execute(
                "UPDATE players SET role = 'multiple' WHERE player_id = ?",
                (player_id,),
            )
        return player_id

    # ------------------------------------------------------------------
    # Roster reload
    # ------------------------------------------------------------------

    def reload_roster(self):
        """
        Clear the in-memory roster cache and re-read scripts/gt_roster.csv.
        Also retroactively purges any opponent-only player rows already stored
        (NULL-ing their FK references first to satisfy foreign key constraints).
        """
        if hasattr(self, "_gt_names"):
            del self._gt_names
        gt = self._load_gt_roster()
        if not gt:
            return

        with self._connect() as conn:
            all_players = conn.execute(
                "SELECT player_id, player_name FROM players"
            ).fetchall()
            to_delete = [
                pid for pid, pname in all_players
                if pname.strip().lower() not in gt
            ]
            if to_delete:
                placeholders = ",".join("?" * len(to_delete))
                conn.execute(
                    f"UPDATE pitches SET pitcher_id = NULL "
                    f"WHERE pitcher_id IN ({placeholders})", to_delete
                )
                conn.execute(
                    f"UPDATE pitches SET batter_id = NULL "
                    f"WHERE batter_id IN ({placeholders})", to_delete
                )
                conn.execute(
                    f"UPDATE fielding SET fielder_id = NULL "
                    f"WHERE fielder_id IN ({placeholders})", to_delete
                )
                conn.execute(
                    f"DELETE FROM players WHERE player_id IN ({placeholders})",
                    to_delete,
                )
                logger.info("reload_roster: purged %d non-GT player rows.", len(to_delete))

        # Bust cache so next call re-reads from disk
        if hasattr(self, "_gt_names"):
            del self._gt_names
        self._load_gt_roster()

    # ------------------------------------------------------------------
    # Game helpers
    # ------------------------------------------------------------------

    def game_exists(self, game_label: str) -> bool:
        """Return True if this game label is already in the DB."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT game_id FROM games WHERE game_label = ?",
                (_clean_str(game_label),),
            ).fetchone()
        return row is not None

    def _create_game(
        self,
        conn: sqlite3.Connection,
        game_label: str,
        file_name: str = None,
    ) -> int:
        game_label = _clean_str(game_label)
        file_name = _clean_str(file_name) if file_name else None
        conn.execute(
            "INSERT OR IGNORE INTO games (game_label, file_name) VALUES (?, ?)",
            (game_label, file_name),
        )
        row = conn.execute(
            "SELECT game_id FROM games WHERE game_label = ?",
            (game_label,),
        ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Ingest — public API
    # ------------------------------------------------------------------

    def ingest_csv(
        self,
        filepath: str | Path,
        game_label: str,
        skip_if_exists: bool = True,
    ) -> Tuple[int, str, str]:
        """
        Read a CSV (or Parquet) file and store it in the DB.
        Returns: (rows_inserted, status, message)
        status in {"success", "warning", "error"}
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return 0, "error", f"File not found: {filepath}"

        if filepath.suffix.lower() in (".parquet", ".parq"):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        if skip_if_exists and self.game_exists(game_label):
            return 0, "warning", f"⚠️ Game label already exists: '{game_label}'. Not added."

        dup_reason = self._duplicate_reason(df, game_label, filepath.name)
        if dup_reason is not None:
            return 0, "warning", dup_reason

        try:
            n = self._ingest_dataframe(df, game_label, filepath.name)
            return n, "success", f"✅ Ingested '{game_label}': {n} rows inserted."
        except Exception as e:
            logger.exception("ingest_csv failed: %s", e)
            return 0, "error", f"❌ Failed to ingest '{game_label}': {e}"

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        game_label: str,
        file_name: str = None,
        skip_if_exists: bool = True,
        force: bool = False,
    ) -> Dict:
        """
        Ingest an already-loaded DataFrame (e.g. from the Streamlit dashboard).
        Returns a dict: {"status": "inserted"|"skipped"|"duplicate", "rows": N,
                         "matching_game": str (only on duplicate)}
        """
        if not force:
            if skip_if_exists and self.game_exists(game_label):
                return {"status": "skipped", "rows": 0}

            dup_reason = self._duplicate_reason(df, game_label, file_name)
            if dup_reason is not None:
                # Extract matching game label for the UI warning
                matching = game_label  # fallback
                with self._connect() as conn:
                    content_hash = self._stable_df_hash(df)
                    row = conn.execute(
                        "SELECT game_label FROM ingested_files WHERE content_sha256 = ?",
                        (content_hash,),
                    ).fetchone()
                    if row:
                        matching = row[0]
                return {"status": "duplicate", "rows": 0, "matching_game": matching}

        try:
            n = self._ingest_dataframe(df, game_label, file_name)
            return {"status": "inserted", "rows": n}
        except Exception as e:
            logger.exception("ingest_dataframe failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Ingest — core logic
    # ------------------------------------------------------------------

    def _ingest_dataframe(
        self,
        df: pd.DataFrame,
        game_label: str,
        file_name: str = None,
    ) -> int:
        """Core ingest logic. Assumes duplicate checks already passed."""
        df = self._normalise_columns(df)

        content_hash = self._stable_df_hash(df)
        columns_sig = _columns_signature(df)
        row_count = int(len(df))

        rows_inserted = 0
        with self._connect() as conn:
            game_id = self._create_game(conn, game_label, file_name)

            conn.execute(
                """
                INSERT INTO ingested_files
                    (game_label, file_name, content_sha256, columns_sig, row_count)
                VALUES (?,?,?,?,?)
                """,
                (
                    _clean_str(game_label),
                    _clean_str(file_name) if file_name else None,
                    content_hash,
                    columns_sig,
                    row_count,
                ),
            )

            for _, row in df.iterrows():
                # Players — GT only; opponents get NULL
                pitcher_id = self._get_or_create_player(
                    conn, row.get("PitcherName"), role="pitcher"
                )
                batter_id = self._get_or_create_player(
                    conn, row.get("BatterName"), role="batter"
                )

                raw_bip = row.get("BallInPlay", row.get("ball_in_play", False))
                bip = _bool_to_int(raw_bip if pd.notna(raw_bip) else False)

                raw_outcome = row.get("PitchOutcome")
                if raw_outcome and str(raw_outcome).strip() not in ("", "nan", "None", "NaN"):
                    pitch_outcome = str(raw_outcome).strip()
                else:
                    pitch_outcome = _categorize_pitch_outcome(_safe(row.get("Result")))

                raw_ev = _safe(row.get("ExitVelo"))
                if bip == 0 and raw_ev is not None:
                    bip = 1
                raw_la = _safe(row.get("LaunchAng"))
                hit_quality = (
                    _calculate_hit_quality(raw_ev, raw_la)
                    if raw_ev is not None and raw_la is not None
                    else "Unknown"
                )

                cur = conn.execute(
                    """
                    INSERT INTO pitches
                        (game_id, inning, at_bat, pitcher_id, batter_id,
                         result, pitch_velo, batter_time_first, batter_top,
                         exit_velo, launch_angle, actual_distance,
                         ball_in_play, pitch_outcome, hit_quality)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        game_id,
                        _safe(row.get("Inning")),
                        _safe(row.get("AtBat")),
                        pitcher_id,
                        batter_id,
                        _safe(row.get("Result")),
                        _safe(row.get("PitchVelo")),
                        _safe(row.get("BatterTimeToFirst")),
                        _safe(row.get("BatterTop")),
                        raw_ev,
                        raw_la,
                        _safe(row.get("ActualDistance")),
                        bip,
                        pitch_outcome,
                        hit_quality,
                    ),
                )
                pitch_id = cur.lastrowid
                rows_inserted += 1

                # Fielding row
                is_event = row.get("IsEventPlayer", False)
                try:
                    is_event = bool(is_event) if pd.notna(is_event) else False
                except (TypeError, ValueError):
                    is_event = False

                if is_event:
                    fielder_name = row.get("EventPlayerName")
                    fielder_id = self._get_or_create_player(
                        conn, fielder_name, role="fielder"
                    )
                    conn.execute(
                        """
                        INSERT INTO fielding
                            (pitch_id, game_id, fielder_id, probability,
                             route_efficiency, move, reaction, reaction_angle,
                             transfer, throw, throw_distance, max_speed)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            pitch_id, game_id, fielder_id,
                            _safe(row.get("FielderProbability")),
                            _safe(row.get("FielderRouteEfficiency")),
                            _safe(row.get("FielderMove")),
                            _safe(row.get("FielderReaction")),
                            _safe(row.get("FielderReactionAngle")),
                            _safe(row.get("FielderTransfer")),
                            _safe(row.get("FielderThrow")),
                            _safe(row.get("FielderThrowDistance")),
                            _safe(row.get("FielderMaxSpeed")),
                        ),
                    )

                # Baserunning row
                has_runner = (
                    pd.notna(row.get("BaserunnerMaxSpeed"))
                    or pd.notna(row.get("BaserunnerInitial"))
                )
                if has_runner:
                    conn.execute(
                        """
                        INSERT INTO baserunning
                            (pitch_id, game_id, max_speed,
                             initial_base, secondary, final_base)
                        VALUES (?,?,?,?,?,?)
                        """,
                        (
                            pitch_id, game_id,
                            _safe(row.get("BaserunnerMaxSpeed")),
                            _safe(row.get("BaserunnerInitial")),
                            _safe(row.get("BaserunnerSecondary")),
                            _safe(row.get("BaserunnerFinal")),
                        ),
                    )

        logger.info("Ingested '%s': %d rows inserted.", game_label, rows_inserted)
        return rows_inserted

    # ------------------------------------------------------------------
    # Column normaliser
    # ------------------------------------------------------------------

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map alternate column names to the canonical CSV column names."""
        df = df.copy()
        alias_map = {
            "PitcherName":           ["pitcher_name", "pitcher", "pitchername"],
            "BatterName":            ["batter_name", "batter", "battername"],
            "Result":                ["result", "play_result", "outcome", "ffx_play_result"],
            "PitchVelo":             ["pitch_velo", "pitch_velocity", "throw_velo", "velo"],
            "BatterTimeToFirst":     ["batter_time_to_first", "time_to_first"],
            "BatterTop":             ["batter_top"],
            "ExitVelo":              ["exit_velo", "exit_velocity", "hit_velo"],
            "LaunchAng":             ["launch_angle", "launch_ang"],
            "ActualDistance":        ["actual_distance"],
            "BallInPlay":            ["ball_in_play", "inplay", "is_bip", "in_play"],
            "PitchOutcome":          ["pitch_outcome", "pitchoutcome"],
            "BaserunnerMaxSpeed":    ["baserunner_max_speed", "max_speed_runner"],
            "BaserunnerInitial":     ["baserunner_initial", "start_base", "initial_base"],
            "BaserunnerSecondary":   ["baserunner_secondary", "secondary_lead"],
            "BaserunnerFinal":       ["baserunner_final", "final_base"],
            "IsEventPlayer":         ["is_event_player", "iseventplayer"],
            "EventPlayerName":       ["event_player_name", "primary_fielder"],
            "FielderProbability":    ["fielder_probability", "probability", "catch_probability"],
            "FielderRouteEfficiency":["fielder_route_efficiency", "route_efficiency"],
            "FielderMove":           ["fielder_move"],
            "FielderReaction":       ["fielder_reaction", "reaction_time", "reaction"],
            "FielderReactionAngle":  ["fielder_reaction_angle", "reaction_angle"],
            "FielderTransfer":       ["fielder_transfer", "transfer"],
            "FielderThrow":          ["fielder_throw"],
            "FielderThrowDistance":  ["fielder_throw_distance", "throw_distance"],
            "FielderMaxSpeed":       ["fielder_max_speed", "max_speed_fielder"],
        }
        existing_lower = {c.lower(): c for c in df.columns}
        for canonical, aliases in alias_map.items():
            if canonical not in df.columns:
                for alias in aliases:
                    if alias.lower() in existing_lower:
                        df[canonical] = df[existing_lower[alias.lower()]]
                        break
        return df

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_games(self) -> pd.DataFrame:
        """Return a DataFrame of all stored games."""
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM games ORDER BY loaded_at DESC", conn)

    def query_all_games(self) -> pd.DataFrame:
        """
        Return the full dataset across all games, joined back to player names.
        Columns match exactly what the dashboard expects.
        """
        sql = """
        SELECT
            g.game_label                                        AS GameID,
            g.game_label                                        AS game_label,
            g.file_name                                         AS FileName,
            g.file_name                                         AS file_name,

            p.inning                                            AS Inning,
            p.at_bat                                            AS AtBat,
            pit.player_name                                     AS PitcherName,
            bat.player_name                                     AS BatterName,
            p.result                                            AS Result,
            p.result                                            AS ResultRaw,
            p.pitch_velo                                        AS PitchVelo,
            p.batter_time_first                                 AS BatterTimeToFirst,
            p.batter_top                                        AS BatterTop,
            p.exit_velo                                         AS ExitVelo,
            p.launch_angle                                      AS LaunchAng,
            p.actual_distance                                   AS ActualDistance,
            p.ball_in_play                                      AS BallInPlay,
            p.pitch_outcome                                     AS PitchOutcome,
            p.hit_quality                                       AS HitQuality,

            f.probability                                       AS FielderProbability,
            f.route_efficiency                                  AS FielderRouteEfficiency,
            f.move                                              AS FielderMove,
            f.reaction                                          AS FielderReaction,
            f.reaction_angle                                    AS FielderReactionAngle,
            f.transfer                                          AS FielderTransfer,
            f.throw                                             AS FielderThrow,
            f.throw_distance                                    AS FielderThrowDistance,
            f.max_speed                                         AS FielderMaxSpeed,
            fld.player_name                                     AS EventPlayerName,
            CASE WHEN f.pitch_id IS NOT NULL THEN 1 ELSE 0 END AS IsEventPlayer,

            CASE
                WHEN f.route_efficiency IS NOT NULL AND f.reaction IS NOT NULL
                THEN ROUND(
                    (f.route_efficiency * 0.7) +
                    (MAX(0.0, MIN(100.0, 100.0 - (f.reaction * 10.0))) * 0.3),
                    2)
                ELSE NULL
            END                                                 AS FieldingEfficiency,

            b.max_speed                                         AS BaserunnerMaxSpeed,
            b.initial_base                                      AS BaserunnerInitial,
            b.secondary                                         AS BaserunnerSecondary,
            b.final_base                                        AS BaserunnerFinal,

            CASE
                WHEN b.final_base IS NOT NULL AND b.initial_base IS NOT NULL
                THEN (b.final_base - b.initial_base)
                ELSE NULL
            END                                                 AS BaseAdvancement,

            p.pitch_id,
            g.game_id
        FROM pitches p
        JOIN  games    g   ON g.game_id    = p.game_id
        LEFT JOIN players pit  ON pit.player_id = p.pitcher_id
        LEFT JOIN players bat  ON bat.player_id = p.batter_id
        LEFT JOIN fielding f   ON f.pitch_id    = p.pitch_id
        LEFT JOIN players fld  ON fld.player_id = f.fielder_id
        LEFT JOIN baserunning b ON b.pitch_id   = p.pitch_id
        ORDER BY g.game_id, p.inning, p.at_bat
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def query_game(self, game_label: str) -> pd.DataFrame:
        """Return data for a single game (by label)."""
        df = self.query_all_games()
        return df[df["game_label"] == game_label].reset_index(drop=True)

    def query_player(self, player_name: str, role: str = "any") -> pd.DataFrame:
        """Pull all data rows involving a specific GT player."""
        df = self.query_all_games()
        if role == "pitcher":
            return df[df["PitcherName"] == player_name].reset_index(drop=True)
        if role == "batter":
            return df[df["BatterName"] == player_name].reset_index(drop=True)
        if role == "fielder":
            return df[df["EventPlayerName"] == player_name].reset_index(drop=True)
        mask = (
            (df["PitcherName"] == player_name)
            | (df["BatterName"] == player_name)
            | (df["EventPlayerName"] == player_name)
        )
        return df[mask].reset_index(drop=True)

    def query_pitching_stats(self) -> pd.DataFrame:
        """Aggregated pitching stats per GT pitcher across all games."""
        sql = """
        SELECT
            pl.player_name                      AS PitcherName,
            COUNT(*)                            AS TotalPitches,
            ROUND(AVG(p.pitch_velo), 1)         AS AvgVelo,
            ROUND(MAX(p.pitch_velo), 1)         AS MaxVelo,
            ROUND(MIN(p.pitch_velo), 1)         AS MinVelo,
            ROUND(AVG(p.exit_velo), 1)          AS AvgExitVelo,
            COUNT(DISTINCT p.game_id)           AS GamesAppeared,
            ROUND(
                100.0 * SUM(CASE WHEN p.pitch_outcome IN ('Strike','Foul') THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 1
            )                                   AS StrikeRate
        FROM pitches p
        JOIN players pl ON pl.player_id = p.pitcher_id
        GROUP BY p.pitcher_id
        ORDER BY TotalPitches DESC
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def query_batting_stats(self) -> pd.DataFrame:
        """Aggregated batting stats per GT batter across all games."""
        sql = """
        SELECT
            pl.player_name                         AS BatterName,
            COUNT(*)                               AS PlateAppearances,
            SUM(p.ball_in_play)                    AS BallsInPlay,
            ROUND(AVG(p.exit_velo), 1)             AS AvgExitVelo,
            ROUND(MAX(p.exit_velo), 1)             AS MaxExitVelo,
            ROUND(AVG(p.launch_angle), 1)          AS AvgLaunchAngle,
            ROUND(AVG(p.batter_time_first), 2)     AS AvgTimeToFirst,
            ROUND(MIN(p.batter_time_first), 2)     AS BestTimeToFirst,
            COUNT(DISTINCT p.game_id)              AS GamesAppeared
        FROM pitches p
        JOIN players pl ON pl.player_id = p.batter_id
        WHERE p.exit_velo IS NOT NULL
        GROUP BY p.batter_id
        ORDER BY AvgExitVelo DESC
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def query_fielding_stats(self) -> pd.DataFrame:
        """Aggregated fielding stats per GT fielder across all games."""
        sql = """
        SELECT
            pl.player_name                              AS FielderName,
            COUNT(*)                                    AS TotalPlays,
            ROUND(AVG(f.route_efficiency), 1)           AS AvgRouteEfficiency,
            ROUND(AVG(f.reaction), 2)                   AS AvgReactionTime,
            ROUND(MAX(f.max_speed), 1)                  AS MaxSpeed,
            ROUND(AVG(f.probability), 1)                AS AvgCatchProbability,
            COUNT(DISTINCT f.game_id)                   AS GamesAppeared
        FROM fielding f
        JOIN players pl ON pl.player_id = f.fielder_id
        GROUP BY f.fielder_id
        ORDER BY AvgRouteEfficiency DESC
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def query_baserunning_stats(self) -> pd.DataFrame:
        """Aggregated baserunning stats across all games."""
        sql = """
        SELECT
            b.initial_base                       AS StartingBase,
            COUNT(*)                             AS Opportunities,
            ROUND(AVG(b.max_speed), 1)           AS AvgMaxSpeed,
            ROUND(MAX(b.max_speed), 1)           AS TopSpeed,
            ROUND(AVG(b.secondary), 2)           AS AvgSecondaryLead,
            COUNT(DISTINCT b.game_id)            AS GamesWithData
        FROM baserunning b
        WHERE b.max_speed IS NOT NULL
        GROUP BY b.initial_base
        ORDER BY b.initial_base
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def query_historical_trends(self) -> pd.DataFrame:
        """Game-by-game trend data for historical comparison charts."""
        sql = """
        SELECT
            g.game_label,
            g.loaded_at,
            COUNT(p.pitch_id)                   AS TotalPitches,
            ROUND(AVG(p.pitch_velo), 1)         AS AvgPitchVelo,
            ROUND(AVG(p.exit_velo), 1)          AS AvgExitVelo,
            ROUND(AVG(p.launch_angle), 1)       AS AvgLaunchAngle,
            ROUND(AVG(f.route_efficiency), 1)   AS AvgRouteEfficiency,
            ROUND(AVG(f.reaction), 2)           AS AvgReactionTime,
            ROUND(AVG(b.max_speed), 1)          AS AvgRunnerSpeed,
            ROUND(
                100.0 * SUM(CASE WHEN p.pitch_outcome IN ('Strike','Foul') THEN 1 ELSE 0 END)
                / NULLIF(COUNT(p.pitch_id), 0), 1
            )                                   AS StrikeRate
        FROM games g
        LEFT JOIN pitches     p ON p.game_id = g.game_id
        LEFT JOIN fielding    f ON f.game_id = g.game_id
        LEFT JOIN baserunning b ON b.game_id = g.game_id
        GROUP BY g.game_id
        ORDER BY g.game_id
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def list_players(self, role: str = None) -> pd.DataFrame:
        """Return all GT players in the DB, optionally filtered by role."""
        with self._connect() as conn:
            if role:
                return pd.read_sql(
                    "SELECT * FROM players WHERE role = ? OR role = 'multiple' ORDER BY player_name",
                    conn,
                    params=(role,),
                )
            return pd.read_sql("SELECT * FROM players ORDER BY player_name", conn)

    # ------------------------------------------------------------------
    # Bulk ingest
    # ------------------------------------------------------------------

    def ingest_folder(self, folder: str | Path, label_prefix: str = "") -> Dict[str, int]:
        """Ingest every CSV and Parquet file in a folder."""
        folder = Path(folder)
        results: Dict[str, int] = {}
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".csv", ".parquet", ".parq"):
                label = f"{label_prefix}{f.stem}" if label_prefix else f.stem
                try:
                    n, status, msg = self.ingest_csv(f, game_label=label)
                    if status == "warning":
                        logger.info("Skipped %s: %s", f.name, msg)
                    results[f.name] = n
                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", f.name, e)
                    results[f.name] = -1
        return results

    # ------------------------------------------------------------------
    # Delete game
    # ------------------------------------------------------------------

    def delete_game(self, game_label: str) -> bool:
        """Remove a game and all its associated rows from every table."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT game_id FROM games WHERE game_label = ?",
                (_clean_str(game_label),),
            ).fetchone()
            if row is None:
                logger.warning("Game '%s' not found.", game_label)
                return False

            game_id = row[0]
            pitch_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT pitch_id FROM pitches WHERE game_id = ?",
                    (game_id,),
                ).fetchall()
            ]
            if pitch_ids:
                placeholders = ",".join("?" * len(pitch_ids))
                conn.execute(
                    f"DELETE FROM fielding    WHERE pitch_id IN ({placeholders})",
                    pitch_ids,
                )
                conn.execute(
                    f"DELETE FROM baserunning WHERE pitch_id IN ({placeholders})",
                    pitch_ids,
                )

            conn.execute("DELETE FROM pitches WHERE game_id = ?", (game_id,))
            conn.execute("DELETE FROM games   WHERE game_id = ?", (game_id,))
            conn.execute(
                "DELETE FROM ingested_files WHERE game_label = ?",
                (_clean_str(game_label),)
            )

        logger.info("Deleted game '%s' (id=%d).", game_label, game_id)
        return True

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def db_summary(self) -> Dict:
        """Quick health-check: row counts per table."""
        with self._connect() as conn:
            return {
                "games":          conn.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                "players":        conn.execute("SELECT COUNT(*) FROM players").fetchone()[0],
                "pitches":        conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0],
                "fielding":       conn.execute("SELECT COUNT(*) FROM fielding").fetchone()[0],
                "baserunning":    conn.execute("SELECT COUNT(*) FROM baserunning").fetchone()[0],
                "ingested_files": conn.execute("SELECT COUNT(*) FROM ingested_files").fetchone()[0],
                "db_path":        str(self.db_path),
                "db_size_kb":     round(self.db_path.stat().st_size / 1024, 1)
                                  if self.db_path.exists() else 0,
            }


# ---------------------------------------------------------------------------
# Quick smoke-test — run: python dbmanager.py path/to/game.csv "Game Label"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    db = GTBaseballDB("data/gt_baseball.db")

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        label = sys.argv[2] if len(sys.argv) > 2 else path.stem
        n, status, msg = db.ingest_csv(path, game_label=label)
        print(msg)

    print("\n── DB Summary ──────────────────────────────────")
    for k, v in db.db_summary().items():
        print(f"  {k:<15}: {v}")

    print("\n── Games ───────────────────────────────────────")
    print(db.list_games().to_string(index=False))

    print("\n── Pitching Stats ──────────────────────────────")
    print(db.query_pitching_stats().to_string(index=False))

    print("\n── Batting Stats ───────────────────────────────")
    print(db.query_batting_stats().to_string(index=False))

    print("\n── Fielding Stats ──────────────────────────────")
    print(db.query_fielding_stats().to_string(index=False))

    print("\n── Historical Trends ───────────────────────────")
    print(db.query_historical_trends().to_string(index=False))