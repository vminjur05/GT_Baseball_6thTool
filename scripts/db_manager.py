"""
db_manager.py — GT Baseball 6th Tool Database Manager
======================================================
Free, zero-infrastructure SQLite database for storing game CSV/Parquet data.
Eliminates the need to re-upload files every session.

Schema is designed around the real Derived_Data CSV format:
    Inning, AtBat, PitcherName, BatterName, Result, PitchVelo,
    BatterTimeToFirst, BatterTop, ExitVelo, LaunchAng, ActualDistance,
    BaserunnerMaxSpeed, BaserunnerInitial, BaserunnerSecondary, BaserunnerFinal,
    IsEventPlayer, EventPlayerName, FielderProbability, FielderRouteEfficiency,
    FielderMove, FielderReaction, FielderReactionAngle, FielderTransfer,
    FielderThrow, FielderThrowDistance, FielderMaxSpeed

Usage:
    from db_manager import GTBaseballDB
    db = GTBaseballDB()                        # creates data/gt_baseball.db
    db.ingest_csv("csv_data/game1.csv", "GT vs UBC Game 1")
    df = db.query_all_games()
    df = db.query_player("Mason Patel", role="pitcher")
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default DB location — lives inside your project's data/ folder
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
    return 1 if val else 0


def _categorize_pitch_outcome(result) -> str:
    """
    Categorize a raw Result string into a pitch outcome label.
    Mirrors the logic in GTBaseballDataLoader._categorize_pitch_outcome().
    """
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
    """
    Calculate hit quality label from exit velo and launch angle.
    Mirrors GTBaseballDataLoader._calculate_hit_quality().
    """
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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GTBaseballDB:
    """
    SQLite-backed store for GT Baseball 6th Tool data.

    Tables
    ------
    games       — one row per uploaded file / game label
    players     — deduplicated player roster (pitchers + batters + fielders)
    pitches     — core pitch-by-pitch rows  (one per CSV row)
    fielding    — fielding metrics (rows where IsEventPlayer = True)
    baserunning — baserunning metrics (rows with runner data)

    The player_id foreign key links all three detail tables back to the
    players table, so you can pull everything for "Mason Patel" in one query.
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
        conn.execute("PRAGMA journal_mode = WAL")   # safe concurrent reads
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Create all tables if they don't already exist."""
        ddl = """
        -- ── games ───────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS games (
            game_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            game_label  TEXT    UNIQUE NOT NULL,
            file_name   TEXT,
            loaded_at   TEXT DEFAULT (datetime('now'))
        );

        -- ── players ──────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS players (
            player_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT    UNIQUE NOT NULL,
            role        TEXT
        );

        -- ── pitches ──────────────────────────────────────────────────────────
        -- One row per CSV row. Stores all columns the dashboard needs directly,
        -- including derived columns (ball_in_play, pitch_outcome, hit_quality)
        -- so they are available when loading from DB without re-running data_loader.
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

        -- ── fielding ─────────────────────────────────────────────────────────
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

        -- ── baserunning ──────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS baserunning (
            run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            pitch_id      INTEGER NOT NULL REFERENCES pitches(pitch_id),
            game_id       INTEGER NOT NULL REFERENCES games(game_id),
            max_speed     REAL,
            initial_base  REAL,
            secondary     REAL,
            final_base    REAL
        );

        -- ── indexes ──────────────────────────────────────────────────────────
        CREATE INDEX IF NOT EXISTS idx_pitches_game    ON pitches(game_id);
        CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher_id);
        CREATE INDEX IF NOT EXISTS idx_pitches_batter  ON pitches(batter_id);
        CREATE INDEX IF NOT EXISTS idx_fielding_game   ON fielding(game_id);
        CREATE INDEX IF NOT EXISTS idx_fielding_player ON fielding(fielder_id);
        CREATE INDEX IF NOT EXISTS idx_baserun_game    ON baserunning(game_id);
        """
        with self._connect() as conn:
            conn.executescript(ddl)

    # ------------------------------------------------------------------
    # Player helpers
    # ------------------------------------------------------------------

    def _get_or_create_player(self, conn: sqlite3.Connection,
                               name: str, role: str = None) -> Optional[int]:
        """
        Return the player_id for `name`, inserting if new.
        If the player already exists with a different role, upgrades to 'multiple'.
        """
        if not name or str(name).strip() in ("", "nan", "None", "NaN"):
            return None
        name = str(name).strip()

        conn.execute(
            "INSERT OR IGNORE INTO players (player_name, role) VALUES (?, ?)",
            (name, role)
        )

        row = conn.execute(
            "SELECT player_id, role FROM players WHERE player_name = ?", (name,)
        ).fetchone()
        if row is None:
            return None

        player_id, existing_role = row
        if role and existing_role and existing_role != role and existing_role != "multiple":
            conn.execute(
                "UPDATE players SET role = 'multiple' WHERE player_id = ?", (player_id,)
            )

        return player_id

    # ------------------------------------------------------------------
    # Game helpers
    # ------------------------------------------------------------------

    def game_exists(self, game_label: str) -> bool:
        """Return True if this game label is already in the DB."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT game_id FROM games WHERE game_label = ?", (game_label,)
            ).fetchone()
        return row is not None

    def _create_game(self, conn: sqlite3.Connection,
                     game_label: str, file_name: str = None) -> int:
        conn.execute(
            "INSERT OR IGNORE INTO games (game_label, file_name) VALUES (?, ?)",
            (game_label, file_name)
        )
        row = conn.execute(
            "SELECT game_id FROM games WHERE game_label = ?", (game_label,)
        ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_csv(self, filepath: str | Path, game_label: str,
                   skip_if_exists: bool = True) -> int:
        """
        Read a CSV (or Parquet) file and store it in the DB.

        Args:
            filepath:       path to the file
            game_label:     human-readable label, e.g. "GT vs UBC 10/24/25"
            skip_if_exists: if True and game_label already in DB, do nothing

        Returns:
            Number of pitch rows inserted (0 if skipped).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if skip_if_exists and self.game_exists(game_label):
            logger.info("'%s' already in DB — skipping.", game_label)
            return 0

        if filepath.suffix.lower() in (".parquet", ".parq"):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        return self._ingest_dataframe(df, game_label, filepath.name)

    def ingest_dataframe(self, df: pd.DataFrame, game_label: str,
                         file_name: str = None, skip_if_exists: bool = True) -> int:
        """
        Ingest an already-loaded DataFrame (e.g. from the Streamlit dashboard).
        This is the hook called from baseball_dashboard.py after a file upload.
        """
        if skip_if_exists and self.game_exists(game_label):
            logger.info("'%s' already in DB — skipping.", game_label)
            return 0
        return self._ingest_dataframe(df, game_label, file_name)

    def _ingest_dataframe(self, df: pd.DataFrame,
                          game_label: str, file_name: str = None) -> int:
        """Internal ingest — normalises columns then bulk-inserts row by row."""

        df = self._normalise_columns(df)

        rows_inserted = 0
        with self._connect() as conn:
            game_id = self._create_game(conn, game_label, file_name)

            for _, row in df.iterrows():

                # ── players ──────────────────────────────────────────────
                pitcher_id = self._get_or_create_player(
                    conn, row.get("PitcherName"), role="pitcher"
                )
                batter_id = self._get_or_create_player(
                    conn, row.get("BatterName"), role="batter"
                )

                # ── BallInPlay — safely handle NA before bool conversion ──
                raw_bip = row.get("BallInPlay", row.get("ball_in_play", False))
                bip = _bool_to_int(raw_bip if pd.notna(raw_bip) else False)

                # ── PitchOutcome — use stored value if present, else derive ─
                raw_outcome = row.get("PitchOutcome")
                if raw_outcome and str(raw_outcome).strip() not in ("", "nan", "None", "NaN"):
                    pitch_outcome = str(raw_outcome).strip()
                else:
                    pitch_outcome = _categorize_pitch_outcome(_safe(row.get("Result")))

                # ── HitQuality — derive from ExitVelo + LaunchAng ────────
                raw_ev = _safe(row.get("ExitVelo"))
                if bip == 0 and raw_ev is not None:
                    bip = 1
                raw_la = _safe(row.get("LaunchAng"))
                if raw_ev is not None and raw_la is not None:
                    hit_quality = _calculate_hit_quality(raw_ev, raw_la)
                else:
                    hit_quality = "Unknown"

                # ── pitch row ────────────────────────────────────────────
                cur = conn.execute("""
                    INSERT INTO pitches
                        (game_id, inning, at_bat, pitcher_id, batter_id,
                         result, pitch_velo, batter_time_first, batter_top,
                         exit_velo, launch_angle, actual_distance,
                         ball_in_play, pitch_outcome, hit_quality)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
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
                ))
                pitch_id = cur.lastrowid
                rows_inserted += 1

                # ── fielding row (IsEventPlayer = True) ──────────────────
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
                    conn.execute("""
                        INSERT INTO fielding
                            (pitch_id, game_id, fielder_id, probability,
                             route_efficiency, move, reaction, reaction_angle,
                             transfer, throw, throw_distance, max_speed)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
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
                    ))

                # ── baserunning row (runner data present) ─────────────────
                has_runner = pd.notna(row.get("BaserunnerMaxSpeed")) or \
                             pd.notna(row.get("BaserunnerInitial"))
                if has_runner:
                    conn.execute("""
                        INSERT INTO baserunning
                            (pitch_id, game_id, max_speed,
                             initial_base, secondary, final_base)
                        VALUES (?,?,?,?,?,?)
                    """, (
                        pitch_id, game_id,
                        _safe(row.get("BaserunnerMaxSpeed")),
                        _safe(row.get("BaserunnerInitial")),
                        _safe(row.get("BaserunnerSecondary")),
                        _safe(row.get("BaserunnerFinal")),
                    ))

        logger.info("Ingested '%s': %d rows inserted.", game_label, rows_inserted)
        return rows_inserted

    # ------------------------------------------------------------------
    # Column normaliser (handles alternate names from Parquet exports)
    # ------------------------------------------------------------------

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map alternate column names to the canonical CSV column names."""
        df = df.copy()
        alias_map = {
            "PitcherName":            ["pitcher_name", "pitcher", "pitchername"],
            "BatterName":             ["batter_name", "batter", "battername"],
            "Result":                 ["result", "play_result", "outcome", "ffx_play_result"],
            "PitchVelo":              ["pitch_velo", "pitch_velocity", "throw_velo", "velo"],
            "BatterTimeToFirst":      ["batter_time_to_first", "time_to_first"],
            "BatterTop":              ["batter_top"],
            "ExitVelo":               ["exit_velo", "exit_velocity", "hit_velo"],
            "LaunchAng":              ["launch_angle", "launch_ang"],
            "ActualDistance":         ["actual_distance"],
            "BallInPlay":             ["ball_in_play", "inplay", "is_bip", "in_play"],
            "PitchOutcome":           ["pitch_outcome", "pitchoutcome"],
            "BaserunnerMaxSpeed":     ["baserunner_max_speed", "max_speed_runner"],
            "BaserunnerInitial":      ["baserunner_initial", "start_base", "initial_base"],
            "BaserunnerSecondary":    ["baserunner_secondary", "secondary_lead"],
            "BaserunnerFinal":        ["baserunner_final", "final_base"],
            "IsEventPlayer":          ["is_event_player", "iseventplayer"],
            "EventPlayerName":        ["event_player_name", "primary_fielder"],
            "FielderProbability":     ["fielder_probability", "probability", "catch_probability"],
            "FielderRouteEfficiency": ["fielder_route_efficiency", "route_efficiency"],
            "FielderMove":            ["fielder_move"],
            "FielderReaction":        ["fielder_reaction", "reaction_time", "reaction"],
            "FielderReactionAngle":   ["fielder_reaction_angle", "reaction_angle"],
            "FielderTransfer":        ["fielder_transfer", "transfer"],
            "FielderThrow":           ["fielder_throw"],
            "FielderThrowDistance":   ["fielder_throw_distance", "throw_distance"],
            "FielderMaxSpeed":        ["fielder_max_speed", "max_speed_fielder"],
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
    # Query helpers — return DataFrames ready for the dashboard
    # ------------------------------------------------------------------

    def list_games(self) -> pd.DataFrame:
        """Return a DataFrame of all stored games."""
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM games ORDER BY loaded_at DESC", conn)

    def query_all_games(self) -> pd.DataFrame:
        """
        Return the full dataset across all games, joined back to player names.
        Equivalent to re-uploading all CSVs — use this to load the dashboard.

        Columns returned match exactly what the dashboard expects:
          PitchOutcome, HitQuality, BallInPlay     — stored in pitches table
          GameID, FileName                          — from games table
          BaseAdvancement, FieldingEfficiency       — calculated in SQL
          ResultRaw                                 — alias of result
          IsEventPlayer                             — derived via CASE WHEN
        """
        sql = """
        SELECT
            -- game identifiers (GameID matches what dashboard sidebar filter uses)
            g.game_label                                        AS GameID,
            g.game_label                                        AS game_label,
            g.file_name                                         AS FileName,
            g.file_name                                         AS file_name,

            -- core pitch columns
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

            -- fielding columns
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

            -- FieldingEfficiency: mirrors _calculate_fielding_efficiency() in data_loader
            -- (route_eff * 0.7) + (reaction_score * 0.3), reaction_score = 100-(reaction*10)
            CASE
                WHEN f.route_efficiency IS NOT NULL AND f.reaction IS NOT NULL
                THEN ROUND(
                    (f.route_efficiency * 0.7) +
                    (MAX(0.0, MIN(100.0, 100.0 - (f.reaction * 10.0))) * 0.3),
                    2)
                ELSE NULL
            END                                                 AS FieldingEfficiency,

            -- baserunning columns
            b.max_speed                                         AS BaserunnerMaxSpeed,
            b.initial_base                                      AS BaserunnerInitial,
            b.secondary                                         AS BaserunnerSecondary,
            b.final_base                                        AS BaserunnerFinal,

            -- BaseAdvancement: how many bases gained this play
            CASE
                WHEN b.final_base IS NOT NULL AND b.initial_base IS NOT NULL
                THEN (b.final_base - b.initial_base)
                ELSE NULL
            END                                                 AS BaseAdvancement,

            -- raw ids
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

    def query_player(self, player_name: str,
                     role: str = "any") -> pd.DataFrame:
        """
        Pull all data rows involving a specific player.

        Args:
            player_name: exact name as stored (e.g. "Mason Patel")
            role:        "pitcher" | "batter" | "fielder" | "any"
        """
        df = self.query_all_games()

        if role == "pitcher":
            return df[df["PitcherName"] == player_name].reset_index(drop=True)
        if role == "batter":
            return df[df["BatterName"] == player_name].reset_index(drop=True)
        if role == "fielder":
            return df[df["EventPlayerName"] == player_name].reset_index(drop=True)

        # "any" — rows where this person appears in any role
        mask = (
            (df["PitcherName"] == player_name) |
            (df["BatterName"] == player_name) |
            (df["EventPlayerName"] == player_name)
        )
        return df[mask].reset_index(drop=True)

    def query_pitching_stats(self) -> pd.DataFrame:
        """Aggregated pitching stats per pitcher across all games."""
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
        """Aggregated batting stats per batter across all games."""
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
        """Aggregated fielding stats per fielder across all games."""
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
        """Return all players in the DB, optionally filtered by role."""
        with self._connect() as conn:
            if role:
                return pd.read_sql(
                    "SELECT * FROM players WHERE role = ? OR role = 'multiple' ORDER BY player_name",
                    conn, params=(role,)
                )
            return pd.read_sql(
                "SELECT * FROM players ORDER BY player_name", conn
            )

    # ------------------------------------------------------------------
    # Bulk-ingest a whole folder
    # ------------------------------------------------------------------

    def ingest_folder(self, folder: str | Path,
                      label_prefix: str = "") -> Dict[str, int]:
        """
        Ingest every CSV and Parquet file in a folder.

        Args:
            folder:       path to csv_data/ or parquet_data/
            label_prefix: optional prefix added to auto-generated labels

        Returns:
            dict of {filename: rows_inserted}
        """
        folder = Path(folder)
        results = {}
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".csv", ".parquet", ".parq"):
                label = f"{label_prefix}{f.stem}" if label_prefix else f.stem
                try:
                    n = self.ingest_csv(f, game_label=label)
                    results[f.name] = n
                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", f.name, e)
                    results[f.name] = -1
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def delete_game(self, game_label: str) -> bool:
        """
        Remove a game and all its associated rows from every table.
        Safe to call; returns False if the label doesn't exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT game_id FROM games WHERE game_label = ?", (game_label,)
            ).fetchone()
            if row is None:
                logger.warning("Game '%s' not found.", game_label)
                return False

            game_id = row[0]
            pitch_ids = [
                r[0] for r in conn.execute(
                    "SELECT pitch_id FROM pitches WHERE game_id = ?", (game_id,)
                ).fetchall()
            ]
            if pitch_ids:
                placeholders = ",".join("?" * len(pitch_ids))
                conn.execute(f"DELETE FROM fielding    WHERE pitch_id IN ({placeholders})", pitch_ids)
                conn.execute(f"DELETE FROM baserunning WHERE pitch_id IN ({placeholders})", pitch_ids)
            conn.execute("DELETE FROM pitches WHERE game_id = ?", (game_id,))
            conn.execute("DELETE FROM games   WHERE game_id = ?", (game_id,))

        logger.info("Deleted game '%s' (id=%d).", game_label, game_id)
        return True

    def db_summary(self) -> Dict:
        """Quick health-check: row counts per table."""
        with self._connect() as conn:
            return {
                "games":       conn.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                "players":     conn.execute("SELECT COUNT(*) FROM players").fetchone()[0],
                "pitches":     conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0],
                "fielding":    conn.execute("SELECT COUNT(*) FROM fielding").fetchone()[0],
                "baserunning": conn.execute("SELECT COUNT(*) FROM baserunning").fetchone()[0],
                "db_path":     str(self.db_path),
                "db_size_kb":  round(self.db_path.stat().st_size / 1024, 1)
                               if self.db_path.exists() else 0,
            }


# ---------------------------------------------------------------------------
# Quick smoke-test — run: python db_manager.py path/to/game.csv "Game Label"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    db = GTBaseballDB("data/gt_baseball.db")

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        label = sys.argv[2] if len(sys.argv) > 2 else path.stem
        n = db.ingest_csv(path, game_label=label)
        print(f"Inserted {n} rows for '{label}'")

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