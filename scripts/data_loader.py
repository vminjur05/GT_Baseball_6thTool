import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GTBaseballDataLoader:
    """
    Handles loading and processing GT Baseball CSV / Parquet data files.
    Designed for the GT Baseball 6th Tool project data format.
    """

    def __init__(self, data_directory: str = "data"):
        self.data_dir = data_directory
        self.game_data: Optional[pd.DataFrame] = None

        # Column expectations (used for basic validation)
        self.required_columns = [
            "Inning",
            "AtBat",
            "PitcherName",
            "BatterName",
            "Result",
            "PitchVelo",
        ]

    def _read_file(self, filepath: str) -> pd.DataFrame:
        """Read CSV or Parquet based on file extension."""
        path = str(filepath)
        ext = os.path.splitext(path)[1].lower()
        if ext in [".parquet", ".parq"]:
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logger.error("Error reading parquet %s: %s", path, e)
                raise
        else:
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.error("Error reading csv %s: %s", path, e)
                raise

    def load_game_data(self, filepath: str, game_id: str = None) -> pd.DataFrame:
        """
        Load GT Baseball game data from CSV or Parquet file.

        Args:
            filepath: Path to the file
            game_id: Optional game identifier to tag rows

        Returns:
            Processed DataFrame with derived columns
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = self._read_file(filepath)
        # Normalize column names to canonical set so downstream code (dashboard) can rely on them
        df = self._normalize_columns(df)
        logger.info("Loaded %d rows from %s", len(df), filepath)

        if game_id:
            df["GameID"] = game_id
        else:
            # add filename as GameID if not provided
            df["GameID"] = os.path.basename(filepath)

        # Basic validation: ensure required columns exist (create if missing)
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            logger.warning("Missing expected columns: %s. Will create them as NaN/None.", missing)
            for c in missing:
                df[c] = pd.NA

        # Clean and derive
        df = self._clean_data(df)
        df = self._add_derived_columns(df)

        logger.info(
            "Processed data: %d total pitches, %d at-bats",
            len(df),
            int(df["AtBat"].nunique()) if "AtBat" in df.columns else 0,
        )

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types (numeric, bool, strings)."""
        df_clean = df.copy()

        # Numeric columns we commonly expect (non-exhaustive)
        numeric_cols = [
            "Inning",
            "AtBat",
            "PitchVelo",
            "BatterTimeToFirst",
            "BatterTop",
            "ExitVelo",
            "LaunchAng",
            "ActualDistance",
            "BaserunnerMaxSpeed",
            "BaserunnerInitial",
            "BaserunnerSecondary",
            "BaserunnerFinal",
            "FielderProbability",
            "FielderRouteEfficiency",
            "FielderMove",
            "FielderReaction",
            "FielderReactionAngle",
            "FielderTransfer",
            "FielderThrow",
            "FielderThrowDistance",
            "FielderMaxSpeed",
        ]

        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Boolean columns (attempt to coerce)
        bool_cols = ["IsEventPlayer"]
        for col in bool_cols:
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].astype(bool)
                except Exception:
                    df_clean[col] = df_clean[col].notna()

        # String columns - safe strip
        string_cols = ["PitcherName", "BatterName", "Result", "EventPlayerName"]
        for col in string_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).fillna("").str.strip()

        return df_clean

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated columns for downstream analysis.
        Defensive: tolerate different column names/absent columns (common with parquet exports).
        """
        df_derived = df.copy()

        # --- Normalize Result-like columns ---
        # Accept common variants and create a uniform 'Result' column if necessary
        result_candidates = [
            "Result",
            "result",
            "ResultText",
            "PlayResult",
            "Outcome",
            "ffx_play_result",
            "ResultRaw",
        ]
        result_col = None
        for c in result_candidates:
            if c in df_derived.columns:
                result_col = c
                break

        if result_col is None:
            df_derived["Result"] = pd.NA
        elif result_col != "Result":
            df_derived["Result"] = df_derived[result_col]

        # --- Normalize BallInPlay variants ---
        bip_candidates = ["BallInPlay", "ball_in_play", "InPlay", "is_bip", "in_play"]
        bip_col = None
        for c in bip_candidates:
            if c in df_derived.columns:
                bip_col = c
                break
        if bip_col is None:
            # Try to infer from Result text (if present)
            if "Result" in df_derived.columns:
                try:
                    df_derived["BallInPlay"] = df_derived["Result"].str.contains("In Play", na=False)
                except Exception:
                    df_derived["BallInPlay"] = False
            else:
                df_derived["BallInPlay"] = False
        else:
            # coerce to boolean safely
            try:
                df_derived["BallInPlay"] = df_derived[bip_col].astype(bool)
            except Exception:
                df_derived["BallInPlay"] = df_derived[bip_col].notna()

        # --- PitchOutcome: safe creation using _categorize_pitch_outcome ---
        # Use get to avoid KeyError; ensure series exists
        result_series = df_derived.get("Result")
        if result_series is None:
            result_series = pd.Series(["Unknown"] * len(df_derived), index=df_derived.index)
        else:
            result_series = result_series.astype(object).fillna("Unknown")
        # apply categorization
        df_derived["PitchOutcome"] = result_series.apply(self._categorize_pitch_outcome)

        # --- Hit quality (ExitVelo + LaunchAng) ---
        if "ExitVelo" in df_derived.columns and "LaunchAng" in df_derived.columns:
            try:
                df_derived["HitQuality"] = df_derived.apply(self._calculate_hit_quality, axis=1)
            except Exception:
                df_derived["HitQuality"] = pd.NA

        # --- Base advancement if baserunner columns exist ---
        if "BaserunnerInitial" in df_derived.columns and "BaserunnerFinal" in df_derived.columns:
            try:
                df_derived["BaseAdvancement"] = df_derived["BaserunnerFinal"] - df_derived["BaserunnerInitial"]
            except Exception:
                df_derived["BaseAdvancement"] = pd.NA

        # --- Fielding efficiency score ---
        # Many exports use different fielder column names (route efficiency / reaction)
        route_col = None
        reaction_col = None
        for cand in ["FielderRouteEfficiency", "route_efficiency", "FielderRoute", "routeEff"]:
            if cand in df_derived.columns:
                route_col = cand
                break
        for cand in ["FielderReaction", "reaction_time", "fielder_reaction", "reaction"]:
            if cand in df_derived.columns:
                reaction_col = cand
                break
        if route_col and reaction_col:
            # map temporary names so helper can use consistent keys
            df_tmp = df_derived.copy()
            df_tmp["FielderRouteEfficiency"] = pd.to_numeric(df_tmp[route_col], errors="coerce")
            df_tmp["FielderReaction"] = pd.to_numeric(df_tmp[reaction_col], errors="coerce")
            try:
                df_derived["FieldingEfficiency"] = df_tmp.apply(self._calculate_fielding_efficiency, axis=1)
            except Exception:
                df_derived["FieldingEfficiency"] = pd.NA
        elif "FielderRouteEfficiency" in df_derived.columns and "FielderReaction" in df_derived.columns:
            try:
                df_derived["FieldingEfficiency"] = df_derived.apply(self._calculate_fielding_efficiency, axis=1)
            except Exception:
                df_derived["FieldingEfficiency"] = pd.NA

        return df_derived

    def _categorize_pitch_outcome(self, result: Optional[str]) -> str:
        """Categorize pitch outcomes into main types."""
        if result is None or (isinstance(result, float) and np.isnan(result)):
            return "Unknown"

        result_str = str(result).lower()

        if "ball" in result_str and "ball in play" not in result_str:
            return "Ball"
        if "strike" in result_str:
            return "Strike"
        if "in play" in result_str or "inplay" in result_str:
            return "In Play"
        if "foul" in result_str:
            return "Foul"
        if "hit by pitch" in result_str or "hitbypitch" in result_str or "hbp" in result_str:
            return "HBP"
        return "Other"

    def _calculate_hit_quality(self, row) -> str:
        """Calculate hit quality based on exit velo and launch angle."""
        try:
            if pd.isna(row.get("ExitVelo")) or pd.isna(row.get("LaunchAng")):
                return "Unknown"
            exit_velo = float(row["ExitVelo"])
            launch_angle = float(row["LaunchAng"])
        except Exception:
            return "Unknown"

        if exit_velo >= 95 and 8 <= launch_angle <= 32:
            return "Barrel"
        if exit_velo >= 95:
            return "Hard Hit"
        if exit_velo < 60:
            return "Weak Contact"
        return "Medium Contact"

    def _calculate_fielding_efficiency(self, row) -> float:
        """Calculate fielding efficiency score (0-100)."""
        # expects FielderRouteEfficiency and FielderReaction (seconds)
        try:
            route = row.get("FielderRouteEfficiency")
            reaction = row.get("FielderReaction")
            if pd.isna(route) or pd.isna(reaction):
                return np.nan
            route_score = float(route)
            reaction_score = 100 - (float(reaction) * 10)  # convert seconds to 0-100 style
            reaction_score = max(0, min(100, reaction_score))
            return (route_score * 0.7) + (reaction_score * 0.3)
        except Exception:
            return np.nan

    def get_pitch_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for pitches."""
        if df is None or len(df) == 0:
            return {}

        summary = {
            "total_pitches": int(len(df)),
            "total_at_bats": int(df["AtBat"].nunique()) if "AtBat" in df.columns else 0,
            "avg_pitch_velocity": float(df["PitchVelo"].mean()) if "PitchVelo" in df.columns else None,
            "pitch_outcomes": df["PitchOutcome"].value_counts().to_dict() if "PitchOutcome" in df.columns else {},
            "pitchers": df["PitcherName"].dropna().unique().tolist() if "PitcherName" in df.columns else [],
            "batters": df["BatterName"].dropna().unique().tolist() if "BatterName" in df.columns else [],
        }

        # Balls in play details
        if "BallInPlay" in df.columns:
            bip = df[df["BallInPlay"] == True]
            summary["balls_in_play"] = int(len(bip))
            if "ExitVelo" in bip.columns:
                summary["avg_exit_velocity"] = float(bip["ExitVelo"].mean())
            if "LaunchAng" in bip.columns:
                summary["avg_launch_angle"] = float(bip["LaunchAng"].mean())

        return summary

    def get_player_stats(self, df: pd.DataFrame, player_type: str = "both") -> Dict:
        """
        Get player statistics grouped by PitcherName / BatterName.
        """
        stats = {}

        if player_type in ["pitcher", "both"] and "PitcherName" in df.columns:
            try:
                pitcher_stats = df.groupby("PitcherName").agg(
                    avg_velocity=("PitchVelo", "mean"),
                    velocity_std=("PitchVelo", "std"),
                    total_pitches=("PitchVelo", "count"),
                    strikes=("PitchOutcome", lambda x: (x == "Strike").sum()),
                    balls_in_play=("BallInPlay", "sum"),
                )
                stats["pitchers"] = pitcher_stats.round(2).to_dict("index")
            except Exception:
                stats["pitchers"] = {}

        if player_type in ["batter", "both"] and "BatterName" in df.columns:
            try:
                bip = df[df["BallInPlay"] == True] if "BallInPlay" in df.columns else df
                if len(bip) > 0:
                    batter_stats = bip.groupby("BatterName").agg(
                        avg_exit_velocity=("ExitVelo", "mean"),
                        avg_launch_angle=("LaunchAng", "mean"),
                        avg_distance=("ActualDistance", "mean"),
                        avg_sprint_speed=("BatterTimeToFirst", "mean"),
                    )
                    stats["batters"] = batter_stats.round(2).to_dict("index")
                else:
                    stats["batters"] = {}
            except Exception:
                stats["batters"] = {}

        return stats

    def load_multiple_games(self, file_patterns: List[str]) -> pd.DataFrame:
        """Load and combine multiple game files (paths list)."""
        all_games = []
        for i, path in enumerate(file_patterns):
            if os.path.exists(path):
                try:
                    gdf = self.load_game_data(path, game_id=f"Game_{i+1}")
                    all_games.append(gdf)
                except Exception as e:
                    logger.warning("Skipping %s due to read/process error: %s", path, e)
            else:
                logger.warning("Path does not exist: %s", path)

        if all_games:
            combined = pd.concat(all_games, ignore_index=True)
            logger.info("Combined %d games (%d rows)", len(all_games), len(combined))
            return combined
        else:
            logger.warning("No game files loaded")
            return pd.DataFrame()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map common column name variants to canonical column names expected by the dashboard.
        This prevents KeyError for PitcherName / BatterName etc. when loading parquet/CSV exports.
        """
        df_norm = df.copy()
        # lowercase map for quick lookup of original column names
        cols_lower = {c.lower(): c for c in df_norm.columns}

        # mapping: canonical_name -> list of possible variants (checked in order)
        mapping = {
            "PitcherName": ["pitchername", "pitcher_name", "pitcher", "pitcherName", "pitcherName".lower()],
            "BatterName": ["battername", "batter_name", "batter", "batterName", "batterName".lower()],
            "ExitVelo": ["exitvelo", "exit_velo", "exit_velocity", "exitvelo".lower()],
            "LaunchAng": ["launchang", "launch_angle", "launchangle", "launchang".lower()],
            "Result": ["result", "resulttext", "playresult", "outcome", "ffx_play_result"],
            "AtBat": ["atbat", "at_bat"],
            "Inning": ["inning"],
            "PitchVelo": ["pitchvelo", "pitch_velo", "pitchvelocity", "pitch_speed"],
            "BallInPlay": ["ballinplay", "ball_in_play", "inplay", "is_bip", "in_play"],
        }

        for canonical, candidates in mapping.items():
            found = None
            for cand in candidates:
                if cand in cols_lower:
                    found = cols_lower[cand]
                    break
            if found:
                # only add canonical if not already present or if different original exists
                if canonical not in df_norm.columns:
                    df_norm[canonical] = df_norm[found]
            else:
                # ensure canonical exists (prevent KeyError later)
                if canonical not in df_norm.columns:
                    df_norm[canonical] = pd.NA

        return df_norm

if __name__ == "__main__":
    # quick local smoke test (adjust path as needed)
    loader = GTBaseballDataLoader(".")
    test_path = "parquet_data/sample.parquet"
    if os.path.exists(test_path):
        df = loader.load_game_data(test_path, game_id="SAMPLE")
        print("Loaded rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Summary:", loader.get_pitch_summary(df))
    else:
        logger.info("No sample.parquet found for quick test.")