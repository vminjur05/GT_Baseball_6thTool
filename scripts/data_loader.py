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

        # Try to infer BallInPlay and basic fielding columns from tracking-like exports
        try:
            df = self._infer_bip_and_fielding(df)
        except Exception:
            # keep defensive: don't fail load if inference has issues
            logger.debug("BIP/fielding inference failed for %s", filepath)

        logger.info(
            "Processed data: %d total pitches, %d at-bats",
            len(df),
            int(df["AtBat"].nunique()) if "AtBat" in df.columns else 0,
        )

        return df
    
    def _infer_bip_and_fielding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressively infer BallInPlay and basic fielding columns when canonical names are missing.
        - Mark BallInPlay True when ExitVelo/LaunchAng/ActualDistance/hit_velo present or Result text indicates 'in play'
        - Map common parquet field names like 'primary_fielder' -> EventPlayerName and infer IsEventPlayer
        Works only when canonical column is missing or entirely null for a row (non-destructive).
        """
        df_inf = df.copy()

        # Ensure canonical columns exist
        if "BallInPlay" not in df_inf.columns:
            df_inf["BallInPlay"] = pd.NA
        if "EventPlayerName" not in df_inf.columns:
            df_inf["EventPlayerName"] = pd.NA
        if "IsEventPlayer" not in df_inf.columns:
            df_inf["IsEventPlayer"] = pd.NA

        # Heuristics: if exit velo / launch angle / actual distance exist -> likely ball in play
        has_hit_metrics = (
            df_inf.get("ExitVelo").notna().fillna(False) |
            df_inf.get("LaunchAng").notna().fillna(False) |
            df_inf.get("ActualDistance").notna().fillna(False) |
            df_inf.get("HitVelo", pd.Series([pd.NA]*len(df_inf))).notna().fillna(False)
        )
        # Also treat explicit text markers in Result
        result_text = df_inf.get("Result")
        res_bip = pd.Series([False]*len(df_inf), index=df_inf.index)
        if result_text is not None:
            try:
                res_bip = result_text.astype(str).str.lower().str.contains("in play|inplay|bip", na=False)
            except Exception:
                res_bip = res_bip

        inferred_bip = has_hit_metrics | res_bip
        # Only set BallInPlay where the canonical value is missing/NA/False
        try:
            current_bip = df_inf["BallInPlay"].astype('boolean')
            mask_to_set = (~current_bip.fillna(False)) & inferred_bip
        except Exception:
            mask_to_set = inferred_bip
        if mask_to_set.any():
            df_inf.loc[mask_to_set, "BallInPlay"] = True

        # Map primary_fielder / player_involved /name -> EventPlayerName when available
        for cand in ["primary_fielder", "player_involved", "player_id", "name"]:
            if cand in df_inf.columns and df_inf[cand].notna().any():
                # only fill where EventPlayerName is empty
                mask = df_inf["EventPlayerName"].isna() | (df_inf["EventPlayerName"].astype(str) == "")
                df_inf.loc[mask & df_inf[cand].notna(), "EventPlayerName"] = df_inf.loc[mask & df_inf[cand].notna(), cand].astype(str)
                # stop after first useful mapping
                break

        # Infer IsEventPlayer if EventPlayerName present or if a 'primary_fielder' flag exists
        try:
            df_inf["IsEventPlayer"] = df_inf["IsEventPlayer"].fillna(False) | df_inf["EventPlayerName"].notna() & (df_inf["EventPlayerName"].astype(str) != "")
        except Exception:
            df_inf["IsEventPlayer"] = df_inf["EventPlayerName"].notna() & (df_inf["EventPlayerName"].astype(str) != "")

        # Coerce types for downstream safety
        try:
            df_inf["BallInPlay"] = df_inf["BallInPlay"].astype(bool)
        except Exception:
            df_inf["BallInPlay"] = df_inf["BallInPlay"].notna()

        return df_inf

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
        # Accept common variants and create a uniform 'Result' column if necessary.
        # Also preserve the original raw column in ResultRaw for inspection.
        result_candidates = [
            "Result",
            "result",
            "ResultText",
            "PlayResult",
            "Outcome",
            "ffx_play_result",
            "ResultRaw",
            "play_result",
        ]
        result_col = None
        for c in result_candidates:
            if c in df_derived.columns:
                result_col = c
                break

        # preserve raw result text if any (use first available candidate)
        if result_col is not None:
            df_derived["ResultRaw"] = df_derived[result_col].astype(object).where(df_derived[result_col].notna(), pd.NA)
            # canonical Result uses the found column (stringified)
            if result_col != "Result":
                df_derived["Result"] = df_derived[result_col].astype(object).where(df_derived[result_col].notna(), pd.NA)
            else:
                df_derived["Result"] = df_derived["Result"].astype(object).where(df_derived["Result"].notna(), pd.NA)
        else:
            df_derived["ResultRaw"] = pd.NA
            df_derived["Result"] = pd.NA

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
        # Normalize strings and apply categorization. Keep Unknown when empty.
        result_series = df_derived.get("Result")
        if result_series is None:
            result_series = pd.Series([pd.NA] * len(df_derived), index=df_derived.index)
        # normalize to string safely for matching
        result_norm = result_series.astype(object).fillna("").astype(str).str.strip()
        df_derived["PitchOutcome"] = result_norm.apply(lambda r: self._categorize_pitch_outcome(r if r != "" else None))

        # Fallback: if BallInPlay True but outcome is Unknown/Other, mark as 'In Play'
        try:
            bip_mask = pd.to_numeric(df_derived.get("BallInPlay"), errors="coerce").astype('boolean').fillna(False)
        except Exception:
            bip_mask = df_derived.get("BallInPlay").notna() & (df_derived.get("BallInPlay") != 0)
        if "PitchOutcome" in df_derived.columns:
            fallback_mask = bip_mask & df_derived["PitchOutcome"].isin(["Unknown", "Other"])
            if fallback_mask.any():
                df_derived.loc[fallback_mask, "PitchOutcome"] = "In Play"

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
        if result is None or (isinstance(result, float) and np.isnan(result)) or (str(result).strip() == ""):
            return "Unknown"

        result_str = str(result).lower()

        # common outcomes / tokens
        if any(tok in result_str for tok in ["ball", "bb", "walk"]) and "in play" not in result_str:
            return "Ball"
        if any(tok in result_str for tok in ["strike", "k", "called strike", "swinging strike"]):
            return "Strike"
        if any(tok in result_str for tok in ["in play", "inplay", "bip", "single", "double", "triple", "home run", "homer", "hr", "groundout", "flyout", "ground out", "fly out", "out"]):
            # more granular: single/double/triple/hr if present
            if "home run" in result_str or "homer" in result_str or "hr" in result_str:
                return "Home Run"
            if "triple" in result_str:
                return "Triple"
            if "double" in result_str:
                return "Double"
            if "single" in result_str:
                return "Single"
            if "ground" in result_str:
                return "Groundout"
            if "fly" in result_str:
                return "Flyout"
            return "In Play"
        if "foul" in result_str:
            return "Foul"
        if any(tok in result_str for tok in ["hit by pitch", "hitbypitch", "hbp"]):
            return "HBP"
        if any(tok in result_str for tok in ["sb", "stolen base"]):
            return "Stolen Base"
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
            "PitcherName": ["pitchername", "pitcher_name", "pitcher", "pitcherid", "pitcher_id"],
            "BatterName": ["battername", "batter_name", "batter", "batterid", "batter_id"],
            "ExitVelo": ["exitvelo", "exit_velo", "exit_velocity", "exit_speed", "hit_velo", "hit_velo", "hit_velo".lower()],
            "LaunchAng": ["launchang", "launch_angle", "launchangle", "launch_ang", "launch_angle".lower()],
            "Result": ["result", "resulttext", "playresult", "outcome", "ffx_play_result", "ResultRaw", "resultraw"],
            "AtBat": ["atbat", "at_bat", "at_bat".lower(), "at_bat"],
            "Inning": ["inning"],
            "PitchVelo": ["pitchvelo", "pitch_velo", "pitchvelocity", "pitch_speed", "throw_velo", "pitch_velo".lower()],
            "BallInPlay": ["ballinplay", "ball_in_play", "inplay", "is_bip", "in_play", "is_in_play", "is_in_play".lower()],

            # --- Baserunning canonical names / common variants (parquet sample) ---
            "BaserunnerInitial": [
                "baserunnerinitial", "baserunner_initial", "runner_initial", "from_base",
                "base_start", "start_base", "initial_base", "start_base_primary_lead", "start_base_lead_differential"
            ],
            "BaserunnerSecondary": [
                "baserunnersecondary", "baserunner_secondary", "secondary_lead",
                "start_base_secondary", "start_base_primary_lead", "start_base_lead_differential",
                "baseline_deviation_at_primary_lead", "baseline_deviation_at_release", "baseline_deviation_at_cross"
            ],
            "BaserunnerFinal": [
                "baserunnerfinal", "baserunner_final", "runner_final", "to_base",
                "base_end", "final_base", "next_base_primary_lead"
            ],
            "BaserunnerMaxSpeed": [
                "baserunnermaxspeed", "baserunner_max_speed", "max_speed", "sprint_speed",
                "maxspeed", "runner_speed", "max_speed_fielder", "max_speed_runner", "max_speed_runner"
            ],

            # parquet distance/time fields kept accessible under canonical-ish names
            "dist_start_base_at_release": ["dist_start_base_at_release", "dist_start_base_at_cross"],
            "dist_next_base_at_release": ["dist_next_base_at_release", "dist_next_base_at_cross"],
            "total_dist_trav_runner": ["total_dist_trav_runner", "total_dist_trav_runner".lower()],
            "total_dist_trav_fielder": ["total_dist_trav_fielder"],

            # batter timing -> canonical used elsewhere
            "BatterTimeToFirst": [
                "battertimetofirst", "batter_time_to_first", "time_to_first_base",
                "time_to_first", "time_to_first_base".lower(), "time_to_first_base"
            ],

            # --- Fielding / event player canonical names / variants (parquet sample) ---
            "EventPlayerName": [
                "eventplayername", "event_player_name", "player_involved", "fielder_name",
                "fielder", "event_player", "name", "primary_fielder"
            ],
            "IsEventPlayer": [
                "iseventplayer", "is_event_player", "event_player_flag", "is_fielder", "is_fielding_play",
                "is_event", "is_eventplayer"
            ],
            "FielderRouteEfficiency": [
                "fielderrouteefficiency", "fielder_route_efficiency", "route_efficiency",
                "routeeff", "route_eff", "route_eff"
            ],
            "FielderReaction": [
                "fielderreaction", "fielder_reaction", "reaction_time", "reaction", "time_top_speed_fielder"
            ],
            "FielderProbability": [
                "fielderprobability", "fielder_probability", "catch_probability", "Probability", "probability"
            ],
            "FielderMaxSpeed": [
                "fieldermaxspeed", "fielder_max_speed", "max_speed_fielder", "time_top_speed_fielder", "max_speed_runner", "max_speed"
            ],
            "FielderThrowDistance": [
                "fielderthrowdistance", "fielder_throw_distance", "throw_distance", "throw_dist"
            ],

            # throw/transfer/pop/retrieval fields
            "ThrowVelo": ["throw_velo", "throwvelo", "throw_velo"],
            "ThrowSequenceID": ["throw_sequence_id", "throwsequenceid"],
            "Transfer": ["transfer", "cleantransfer", "CleanTransfer"],
            "OutOfPlay": ["OutOfPlay", "outofplay", "out_of_play"],
            "PopTime": ["pop_time", "poptime"],
            "RetrievalTime": ["retrieval"],

            # hitting fields mapping
            "HitVelo": ["hit_velo", "hitvelo", "hit_velo".lower()],
            "LaunchAng_raw": ["launch_angle", "launchangle", "Launch_Ang", "launch_angle".lower()],
            "SprayAngle": ["spray_angle"],
            "IsRocket": ["is_rocket"],
            "ActualDistance": ["actual_distance", "actualdistance", "ActualDistance"],
            "HangTime": ["hang_time", "hangtime"],

            # raw ids / references (keep available)
            "mlb_game_str": ["mlb_game_str"],
            "ffx_play_guid": ["ffx_play_guid"],
            "ffx_pitch_time": ["ffx_pitch_time"],
            "lead_baserunner": ["lead_baserunner"],

            # helper: keep raw sample columns accessible if downstream code references them
            "route_efficiency": ["route_efficiency"],
            "reaction_angle": ["reaction_angle"],
            "dist_to_play": ["dist_to_play", "distance_to_play"]
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