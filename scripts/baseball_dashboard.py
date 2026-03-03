import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from pathlib import Path
from data_loader import GTBaseballDataLoader
from baseball_analyzer import GTBaseballAnalyzer
from report_generator import ReportGenerator
from db_integration import render_database_tab, auto_save_to_db, render_duplicate_warning


# ---------------------------------------------------------------------------
# GT Roster helper — loaded once, cached, used to filter sidebar dropdowns
# ---------------------------------------------------------------------------

@st.cache_data
def _load_gt_roster_names() -> set:
    """
    Return a set of lowercase GT player names from scripts/gt_roster.csv.
    Falls back to empty set (shows all players) if the file is missing.
    """
    roster_path = Path("scripts/gt_roster.csv")
    if not roster_path.exists():
        return set()
    try:
        df = pd.read_csv(roster_path)
        col = None
        for c in ["player_name", "name", "PlayerName", "full_name"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            col = df.columns[0]
        return set(df[col].dropna().astype(str).str.strip().str.lower())
    except Exception:
        return set()


def _filter_to_gt(names: list, gt_names: set) -> list:
    """Return only names that appear in the GT roster (case-insensitive)."""
    if not gt_names:
        return names  # no roster loaded → show everyone (safe degradation)
    return [n for n in names if str(n).strip().lower() in gt_names]


class GTBaseballDashboard:
    """
    Interactive Streamlit dashboard for GT Baseball 6th Tool analytics.
    """

    def __init__(self):
        self.setup_page_config()

    # -----------------------------------------------------------------------
    # Column helpers
    # -----------------------------------------------------------------------

    def _find_col(self, df, candidates):
        """Return first existing column name from candidates, or None."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _numeric_series(self, df, candidates):
        """Return numeric Series for first matching candidate; else NaN series."""
        col = self._find_col(df, candidates)
        if col is None:
            return pd.Series([pd.NA] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    def _name_series(self, df, candidates):
        """Return string Series for first matching candidate; else empty strings."""
        col = self._find_col(df, candidates)
        if col is None:
            return pd.Series([""] * len(df), index=df.index)
        return df[col].astype(str).fillna("")

    # -----------------------------------------------------------------------
    # Page config — dark mode + GT brand colours
    # -----------------------------------------------------------------------

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="GT Baseball 6th Tool Dashboard",
            page_icon="⚾",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown("""
        <style>
        /* ── GT brand colours ─────────────────────────────────────────── */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #B3A369;
            text-align: center;
            padding: 1rem 0 0.5rem;
            border-bottom: 3px solid #B3A369;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background-color: #1e2530 !important;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #B3A369;
        }
        /* ── Buttons ──────────────────────────────────────────────────── */
        .stButton > button {
            background-color: #003057 !important;
            color: #ffffff !important;
            border: 1px solid #B3A369 !important;
            border-radius: 6px !important;
        }
        .stButton > button:hover {
            background-color: #B3A369 !important;
            border-color: #B3A369 !important;
            color: #ffffff !important;
        }
        /* ── Tabs ─────────────────────────────────────────────────────── */
        .stTabs [aria-selected="true"] {
            background-color: #003057 !important;
            color: #B3A369 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_data(self):
        """Load and cache data."""
        if "game_data" not in st.session_state:
            try:
                from db_integration import _get_db
                db = _get_db()
                summary = db.db_summary()
                if summary["games"] > 0:
                    st.info(
                        f"📦 Database has {summary['games']} games "
                        f"and {summary['pitches']} pitches stored."
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load All Games from Database"):
                            st.session_state.game_data = db.query_all_games()
                            st.rerun()
                    with col2:
                        games_df = db.list_games()
                        chosen = st.selectbox(
                            "Or load one game:",
                            ["— pick one —"] + games_df["game_label"].tolist(),
                        )
                        if chosen != "— pick one —":
                            st.session_state.game_data = db.query_game(chosen)
                            st.rerun()
                    st.divider()
            except Exception:
                st.warning("⚠️ Database not available or inaccessible.")

            loader = GTBaseballDataLoader()

            uploaded_files = st.file_uploader(
                "Upload GT Baseball CSV/Parquet files",
                accept_multiple_files=True,
                type=["csv", "parquet"],
            )

            if uploaded_files:
                all_data = []
                for i, file in enumerate(uploaded_files):
                    try:
                        if file.name.lower().endswith(".parquet"):
                            import tempfile, os
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                            tmp.write(file.getvalue())
                            tmp.flush()
                            df = pd.read_parquet(tmp.name)
                            tmp.close()
                            os.unlink(tmp.name)
                        else:
                            df = pd.read_csv(file)

                        df = loader._normalize_columns(df)
                        df["GameID"] = f"Game_{i+1}"
                        df["FileName"] = file.name
                        all_data.append(df)
                        auto_save_to_db(df, game_label=file.name, file_name=file.name)
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")

                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    loader.data = combined_data
                    st.session_state.game_data = loader._clean_data(combined_data)
                    st.session_state.game_data = loader._add_derived_columns(
                        st.session_state.game_data
                    )
                    try:
                        st.session_state.game_data = loader._infer_bip_and_fielding(
                            st.session_state.game_data
                        )
                    except Exception:
                        pass

                    bip_count = 0
                    try:
                        bip_count = int(
                            st.session_state.game_data["BallInPlay"].astype(bool).sum()
                        )
                    except Exception:
                        pass
                    fielding_rows = 0
                    try:
                        fielding_rows = int(len(
                            st.session_state.game_data[
                                (st.session_state.game_data.get("IsEventPlayer") == True)
                                & (st.session_state.game_data.get("EventPlayerName").notna())
                            ]
                        ))
                    except Exception:
                        pass

                    if bip_count == 0:
                        st.warning(
                            "No balls in play detected after automatic inference. "
                            "Hitting charts will be empty."
                        )
                        with st.expander("Why no BIP? See columns & sample rows"):
                            st.write("Available columns (first 120):")
                            st.write(st.session_state.game_data.columns.tolist()[:120])
                            st.write("Sample rows:")
                            st.dataframe(
                                st.session_state.game_data.head(5), use_container_width=True
                            )

                    if fielding_rows == 0:
                        st.info(
                            "No fielding/event-player rows detected. "
                            "Defensive analytics will be limited."
                        )
                        with st.expander("Why no fielding data? See candidate columns"):
                            st.write(
                                "Look for: 'primary_fielder', 'player_involved', "
                                "'route_efficiency', 'reaction_time', 'Probability'"
                            )
                            st.write(st.session_state.game_data.columns.tolist()[:120])
                            st.dataframe(
                                st.session_state.game_data.head(5), use_container_width=True
                            )

                    st.success(
                        f"Loaded {len(all_data)} game(s) with "
                        f"{len(st.session_state.game_data):,} total pitches"
                    )
                    return st.session_state.game_data
            else:
                st.info("Please upload CSV files to begin analysis")
                return None

        return st.session_state.game_data

    # -----------------------------------------------------------------------
    # Sidebar — ALL filters live here; dropdowns show GT players only
    # -----------------------------------------------------------------------

    def render_sidebar(self, data):
        """Render sidebar with filters and controls."""
        st.sidebar.header("🎛️ Analysis Controls")

        # Show duplicate-upload warning banner if one is pending
        render_duplicate_warning()

        # ── Game selection ───────────────────────────────────────────────
        if "GameID" in data.columns:
            games = ["All Games"] + list(data["GameID"].unique())
            selected_game = st.sidebar.selectbox("Select Game:", games)
            if selected_game != "All Games":
                data = data[data["GameID"] == selected_game]

        st.sidebar.subheader("Player Filters")

        # Load GT roster names (cached after first call)
        gt_names = _load_gt_roster_names()

        # ── Pitcher filter — GT pitchers only ────────────────────────────
        pitcher_col = self._find_col(
            data, ["PitcherName", "pitcher", "pitcher_name", "pitcher_id"]
        )
        all_pitchers_raw = (
            list(data[pitcher_col].dropna().unique()) if pitcher_col else []
        )
        gt_pitchers = sorted(_filter_to_gt(all_pitchers_raw, gt_names))
        selected_pitcher = st.sidebar.selectbox(
            "Pitcher (GT only):", ["All Pitchers"] + gt_pitchers
        )

        # ── Batter filter — GT batters only ──────────────────────────────
        batter_col = self._find_col(
            data, ["BatterName", "batter", "batter_name", "batter_id"]
        )
        all_batters_raw = (
            list(data[batter_col].dropna().unique()) if batter_col else []
        )
        gt_batters = sorted(_filter_to_gt(all_batters_raw, gt_names))
        selected_batter = st.sidebar.selectbox(
            "Batter (GT only):", ["All Batters"] + gt_batters
        )

        # ── Apply filters ─────────────────────────────────────────────────
        filtered = data.copy()
        if selected_pitcher != "All Pitchers" and pitcher_col:
            filtered = filtered[filtered[pitcher_col] == selected_pitcher]
        if selected_batter != "All Batters" and batter_col:
            filtered = filtered[filtered[batter_col] == selected_batter]

        # ── Roster status note ────────────────────────────────────────────
        if gt_names:
            st.sidebar.caption(
                f"🟡 GT roster loaded — {len(gt_names)} players. "
                "Edit `scripts/gt_roster.csv` to update."
            )
        else:
            st.sidebar.caption(
                "⚠️ `scripts/gt_roster.csv` not found — showing all players. "
                "Add the file to enable GT-only filtering."
            )

        return filtered, selected_pitcher, selected_batter

    # -----------------------------------------------------------------------
    # Overview metrics
    # -----------------------------------------------------------------------

    def render_overview_metrics(self, data):
        """Render key performance metrics."""
        st.markdown(
            '<div class="main-header">⚾ GT Baseball 6th Tool Analytics</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Pitches", f"{len(data):,}")

        with col2:
            pv = self._numeric_series(
                data, ["PitchVelo", "pitch_velo", "pitch_velocity", "throw_velo", "velo"]
            )
            avg_velocity = pv.mean()
            st.metric(
                "Avg Pitch Velocity",
                f"{avg_velocity:.1f} mph" if not pd.isna(avg_velocity) else "N/A",
            )

        with col3:
            bip = self._numeric_series(data, ["BallInPlay", "ball_in_play"])
            try:
                bip_count = int(bip.astype(bool).sum())
            except Exception:
                bip_count = int(bip.dropna().sum()) if bip.dropna().size > 0 else 0
            st.metric("Balls in Play", f"{bip_count:,}")

        with col4:
            ev = self._numeric_series(
                data, ["ExitVelo", "exit_velo", "ExitVelocity", "Exit_Velo"]
            )
            bip_mask = (
                self._numeric_series(data, ["BallInPlay", "ball_in_play"])
                .fillna(0)
                .astype(bool)
            )
            avg_exit_velo = ev[bip_mask].mean() if bip_mask.any() else float("nan")
            st.metric(
                "Avg Exit Velocity",
                f"{avg_exit_velo:.1f} mph" if not pd.isna(avg_exit_velo) else "N/A",
            )

        with col5:
            gt_names = _load_gt_roster_names()
            pser = self._name_series(data, ["PitcherName", "pitcher", "pitcher_name"])
            bser = self._name_series(data, ["BatterName", "batter", "batter_name"])
            all_names = set(pser.dropna().unique()) | set(bser.dropna().unique())
            if gt_names:
                gt_count = sum(1 for n in all_names if str(n).lower() in gt_names)
                st.metric("GT Players Active", gt_count)
            else:
                st.metric("Total Players", len(all_names))

    # -----------------------------------------------------------------------
    # Pitching analysis
    # -----------------------------------------------------------------------

    def render_pitching_analysis(self, data):
        """Render pitching analysis section."""
        st.header("🥎 Pitching Analysis")

        pv = self._numeric_series(
            data, ["PitchVelo", "pitch_velo", "pitch_velocity", "throw_velo", "velo"]
        )
        temp = data.copy()
        temp["_PitchVelo"] = pv

        if temp["_PitchVelo"].dropna().empty:
            st.warning("No pitch velocity data available for pitching analysis.")
            if "PitchOutcome" in data.columns:
                outcome_counts = data["PitchOutcome"].value_counts()
                st.plotly_chart(
                    px.pie(
                        values=outcome_counts.values,
                        names=outcome_counts.index,
                        title="Pitch Outcome Distribution",
                    ),
                    use_container_width=True,
                )
            return

        col1, col2 = st.columns(2)

        with col1:
            fig_vel = px.histogram(
                temp.dropna(subset=["_PitchVelo"]),
                x="_PitchVelo",
                nbins=20,
                title="Pitch Velocity Distribution",
                labels={"_PitchVelo": "Velocity (mph)", "count": "Frequency"},
            )
            mean_vel = temp["_PitchVelo"].mean()
            fig_vel.add_vline(
                x=mean_vel, line_dash="dash", annotation_text=f"Mean: {mean_vel:.1f}"
            )
            st.plotly_chart(fig_vel, use_container_width=True)

        with col2:
            if "PitchOutcome" in data.columns:
                outcome_counts = data["PitchOutcome"].value_counts()
                st.plotly_chart(
                    px.pie(
                        values=outcome_counts.values,
                        names=outcome_counts.index,
                        title="Pitch Outcome Distribution",
                    ),
                    use_container_width=True,
                )

        if "Inning" in temp.columns and temp["Inning"].nunique() > 1:
            st.subheader("Velocity by Inning")
            df_box = temp.dropna(subset=["_PitchVelo", "Inning"])
            if len(df_box) > 0:
                st.plotly_chart(
                    px.box(
                        df_box,
                        x="Inning",
                        y="_PitchVelo",
                        title="Pitch Velocity by Inning",
                        labels={"_PitchVelo": "Velocity (mph)"},
                    ),
                    use_container_width=True,
                )

        st.subheader("Pitcher Comparison")
        pitcher_col = self._find_col(
            temp, ["PitcherName", "pitcher", "pitcher_name", "player_name", "name"]
        )
        if pitcher_col:
            pv_by_pitcher = (
                temp.dropna(subset=[pitcher_col])
                .groupby(pitcher_col)["_PitchVelo"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "Avg Velocity", "count": "Total Pitches"})
            )
            if "PitchOutcome" in temp.columns:
                strike_rate = temp.groupby(pitcher_col)["PitchOutcome"].apply(
                    lambda s: (s.isin(["Strike", "Foul"]).sum() / max(1, len(s))) * 100
                )
                pv_by_pitcher["Strike Rate %"] = strike_rate
            else:
                pv_by_pitcher["Strike Rate %"] = pd.NA
            st.dataframe(pv_by_pitcher.round(2), use_container_width=True)
        else:
            st.info("No pitcher identifier column found for pitcher comparison.")

    # -----------------------------------------------------------------------
    # Hitting analysis
    # -----------------------------------------------------------------------

    def render_hitting_analysis(self, data):
        """Render hitting analysis section."""
        st.header("🏏 Hitting Analysis")

        hit_data = data[data["BallInPlay"].fillna(False).astype(bool)]

        if len(hit_data) == 0:
            st.warning("No balls in play data available for hitting analysis.")
            return

        col1, col2 = st.columns(2)

        with col1:
            if "ExitVelo" in hit_data.columns:
                fig_exit = px.histogram(
                    hit_data.dropna(subset=["ExitVelo"]),
                    x="ExitVelo",
                    nbins=15,
                    title="Exit Velocity Distribution",
                    labels={"ExitVelo": "Exit Velocity (mph)"},
                )
                fig_exit.add_vline(
                    x=95, line_dash="dash", line_color="red",
                    annotation_text="Hard Hit Threshold"
                )
                st.plotly_chart(fig_exit, use_container_width=True)

        with col2:
            if "LaunchAng" in hit_data.columns:
                st.plotly_chart(
                    px.histogram(
                        hit_data.dropna(subset=["LaunchAng"]),
                        x="LaunchAng",
                        nbins=15,
                        title="Launch Angle Distribution",
                        labels={"LaunchAng": "Launch Angle (degrees)"},
                    ),
                    use_container_width=True,
                )

        if "ExitVelo" in hit_data.columns and "LaunchAng" in hit_data.columns:
            st.subheader("Exit Velocity vs Launch Angle")
            scatter_data = hit_data.dropna(subset=["ExitVelo", "LaunchAng"])
            if len(scatter_data) > 0:
                fig_scatter = px.scatter(
                    scatter_data,
                    x="LaunchAng",
                    y="ExitVelo",
                    color="Result",
                    title="Exit Velocity vs Launch Angle",
                    labels={
                        "LaunchAng": "Launch Angle (degrees)",
                        "ExitVelo": "Exit Velocity (mph)",
                    },
                )
                fig_scatter.add_hrect(
                    y0=95, y1=scatter_data["ExitVelo"].max(),
                    fillcolor="green", opacity=0.2, annotation_text="Hard Hit Zone"
                )
                fig_scatter.add_vrect(
                    x0=8, x1=32, fillcolor="red", opacity=0.2, annotation_text="Barrel Zone"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        if "BatterTimeToFirst" in hit_data.columns:
            sprint_data = hit_data.dropna(subset=["BatterTimeToFirst"])
            if len(sprint_data) > 0:
                st.subheader("Sprint Speed Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        px.histogram(
                            sprint_data,
                            x="BatterTimeToFirst",
                            title="Time to First Base Distribution",
                            labels={"BatterTimeToFirst": "Time to First (seconds)"},
                        ),
                        use_container_width=True,
                    )
                with col2:
                    st.subheader("Fastest Times to First")
                    st.dataframe(
                        sprint_data.nsmallest(10, "BatterTimeToFirst")[
                            ["BatterName", "BatterTimeToFirst"]
                        ],
                        use_container_width=True,
                    )

    # -----------------------------------------------------------------------
    # Fielding analysis
    # -----------------------------------------------------------------------

    def render_fielding_analysis(self, data):
        """Render fielding analysis section."""
        st.header("🥅 Fielding Analysis")

        field_data = data[
            (data["IsEventPlayer"] == True) & (data["EventPlayerName"].notna())
        ]

        if len(field_data) == 0:
            st.warning("No fielding data available for analysis.")
            return

        col1, col2 = st.columns(2)

        with col1:
            if "FielderRouteEfficiency" in field_data.columns:
                st.plotly_chart(
                    px.histogram(
                        field_data.dropna(subset=["FielderRouteEfficiency"]),
                        x="FielderRouteEfficiency",
                        title="Route Efficiency Distribution",
                        labels={"FielderRouteEfficiency": "Route Efficiency (%)"},
                    ),
                    use_container_width=True,
                )

        with col2:
            if "FielderReaction" in field_data.columns:
                st.plotly_chart(
                    px.histogram(
                        field_data.dropna(subset=["FielderReaction"]),
                        x="FielderReaction",
                        title="Reaction Time Distribution",
                        labels={"FielderReaction": "Reaction Time (seconds)"},
                    ),
                    use_container_width=True,
                )

        if (
            "FielderRouteEfficiency" in field_data.columns
            and "FielderReaction" in field_data.columns
        ):
            st.subheader("Fielder Performance Comparison")
            fielder_stats = (
                field_data.groupby("EventPlayerName")
                .agg({
                    "FielderRouteEfficiency": "mean",
                    "FielderReaction": "mean",
                    "FielderMaxSpeed": "max",
                })
                .round(2)
            )
            fielder_stats.columns = ["Avg Route Efficiency", "Avg Reaction Time", "Max Speed"]
            st.dataframe(fielder_stats, use_container_width=True)

        if "FielderProbability" in field_data.columns:
            prob_data = field_data.dropna(subset=["FielderProbability"])
            if len(prob_data) > 0:
                st.subheader("Catch Probability Analysis")
                st.plotly_chart(
                    px.scatter(
                        prob_data,
                        x="FielderProbability",
                        y="FielderRouteEfficiency",
                        color="EventPlayerName",
                        title="Route Efficiency vs Catch Probability",
                        labels={
                            "FielderProbability": "Catch Probability (%)",
                            "FielderRouteEfficiency": "Route Efficiency (%)",
                        },
                    ),
                    use_container_width=True,
                )

    # -----------------------------------------------------------------------
    # Baserunning analysis
    # -----------------------------------------------------------------------

    def render_baserunning_analysis(self, data):
        """Render baserunning analysis section."""
        st.header("🏃 Baserunning Analysis")

        speed_col = self._find_col(
            data,
            ["BaserunnerMaxSpeed", "baserunner_max_speed", "max_speed",
             "MaxSpeed", "runner_speed", "speed"],
        )

        if speed_col is None:
            st.warning("No baserunning speed data available for analysis.")
            return

        base_data = data[data[speed_col].notna()]

        if len(base_data) == 0:
            st.warning("No baserunning data available for analysis.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                px.histogram(
                    base_data,
                    x=speed_col,
                    title="Baserunner Max Speed Distribution",
                    labels={speed_col: "Max Speed (mph)"},
                ),
                use_container_width=True,
            )

        with col2:
            initial_col = self._find_col(
                base_data,
                ["BaserunnerInitial", "baserunner_initial", "base_start",
                 "initial_base", "from_base"],
            )
            if initial_col:
                base_data_clean = base_data.dropna(subset=[initial_col])
                if len(base_data_clean) > 0:
                    st.plotly_chart(
                        px.box(
                            base_data_clean,
                            x=initial_col,
                            y=speed_col,
                            title="Speed by Starting Base",
                            labels={
                                initial_col: "Starting Base",
                                speed_col: "Max Speed (mph)",
                            },
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("Insufficient data for base-by-base speed analysis.")
            else:
                st.info("Base position data not available for detailed analysis.")

    # -----------------------------------------------------------------------
    # Game flow
    # -----------------------------------------------------------------------

    def render_game_flow(self, data):
        """Render game flow analysis."""
        st.header("📊 Game Flow Analysis")

        if "Inning" not in data.columns:
            st.warning("No inning data available for game flow analysis.")
            return

        inning_summary = (
            data.groupby("Inning")
            .agg({"PitchVelo": "mean", "AtBat": "nunique", "PitchOutcome": "count"})
            .round(2)
        )
        inning_summary.columns = ["Avg Velocity", "At Bats", "Total Pitches"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                x=inning_summary.index,
                y=inning_summary["Total Pitches"],
                title="Pitches per Inning",
            )
            fig.update_xaxes(title="Inning")
            fig.update_yaxes(title="Number of Pitches")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                x=inning_summary.index,
                y=inning_summary["Avg Velocity"],
                title="Average Velocity by Inning",
            )
            fig.update_xaxes(title="Inning")
            fig.update_yaxes(title="Average Velocity (mph)")
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Defensive coaching analysis
    # -----------------------------------------------------------------------

    def render_defensive_coaching_analysis(self, data):
        """Render defensive coaching insights."""
        st.header("🛡️ Defensive Coaching Analytics")

        tracking_file = st.file_uploader(
            "Upload tracking file (CSV or Parquet)", type=["csv", "parquet"]
        )
        expected_file = st.file_uploader(
            "Optional expected positions (CSV or Parquet)", type=["csv", "parquet"]
        )

        tracking_df = None
        expected_df = None
        if tracking_file is not None:
            if tracking_file.name.lower().endswith(".parquet"):
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                tmp.write(tracking_file.getvalue())
                tmp.flush()
                tracking_df = pd.read_parquet(tmp.name)
                tmp.close()
                os.unlink(tmp.name)
            else:
                tracking_df = pd.read_csv(tracking_file)

        if expected_file is not None:
            if expected_file.name.lower().endswith(".parquet"):
                import tempfile, os
                tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                tmp2.write(expected_file.getvalue())
                tmp2.flush()
                expected_df = pd.read_parquet(tmp2.name)
                tmp2.close()
                os.unlink(tmp2.name)
            else:
                expected_df = pd.read_csv(expected_file)

        from defensive_analytics import DefensiveAnalytics
        try:
            defensive_analyzer = DefensiveAnalytics(data)
        except TypeError:
            defensive_analyzer = DefensiveAnalytics()
            if hasattr(defensive_analyzer, "load_data"):
                try:
                    defensive_analyzer.load_data(data)
                except Exception:
                    pass

        def _safe_call(method_name, *args, **kwargs):
            method = getattr(defensive_analyzer, method_name)
            for attempt in [
                lambda: method(*args, **kwargs),
                lambda: method(*args),
                lambda: method(),
            ]:
                try:
                    return attempt()
                except Exception:
                    continue
            return {"error": f"failed to call {method_name}"}

        positioning = _safe_call("analyze_fielder_positioning", tracking_df, expected_df)

        if isinstance(positioning, dict) and "error" in positioning:
            st.error("Positioning analysis failed: " + str(positioning.get("error")))
        else:
            pos_df = pd.DataFrame(positioning).T
            st.dataframe(pos_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    x=pos_df.index,
                    y=pos_df["avg_route_efficiency"],
                    title="Average Route Efficiency by Fielder",
                    labels={"x": "Fielder", "y": "Route Efficiency (%)"},
                )
                fig.add_hline(
                    y=85, line_dash="dash", line_color="green",
                    annotation_text="Target: 85%"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    x=pos_df["avg_reaction_time"],
                    y=pos_df["avg_route_efficiency"],
                    text=pos_df.index,
                    title="Reaction Time vs Route Efficiency",
                    labels={"x": "Avg Reaction Time (s)", "y": "Route Efficiency (%)"},
                )
                fig.add_vline(
                    x=pos_df["avg_reaction_time"].mean(),
                    line_dash="dash", annotation_text="Team Avg"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Defensive Shift Effectiveness")
        shift_analysis = defensive_analyzer.validate_defensive_shifts()
        col1, col2, col3 = st.columns(3)
        col1.metric("High Probability Catches", shift_analysis["high_probability_catches"])
        col2.metric("Missed Opportunities", shift_analysis["missed_opportunities"])
        col3.metric(
            "Optimal Positioning Rate",
            f"{shift_analysis['optimal_positioning_rate']:.1f}%"
        )

        st.subheader("Coaching Recommendations")
        insights = defensive_analyzer.reaction_time_coaching_insights()
        if insights["players_needing_improvement"]:
            st.warning("**Players needing reaction time improvement:**")
            for player in insights["players_needing_improvement"]:
                st.write(f"• {player}")
        if insights["top_performers"]:
            st.success("**Top defensive performers:**")
            for player in insights["top_performers"]:
                st.write(f"• {player}")

        heatmap = defensive_analyzer.create_fielding_heatmap()
        if heatmap:
            st.subheader("Fielding Performance Heatmap")
            st.plotly_chart(heatmap, use_container_width=True)

    # -----------------------------------------------------------------------
    # Coaching reports
    # -----------------------------------------------------------------------

    def render_coaching_reports(self, data):
        """Generate automated coaching reports."""
        st.header("📋 Coaching Reports & Insights")
        st.subheader("Game Summary Generator")

        if st.button("Generate Coaching Summary"):
            summary = self.generate_coaching_summary(data)
            st.text_area("Coaching Summary", summary, height=400)
            st.download_button(
                label="Download Coaching Report",
                data=summary,
                file_name=f"coaching_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )
            try:
                rg = ReportGenerator(data)
                pdf_bytes = rg.export_pdf_bytes()
                st.download_button(
                    label="Download Weekly PDF Report",
                    data=pdf_bytes,
                    file_name=f"weekly_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")

        st.subheader("Top 5 Game Insights")
        for i, insight in enumerate(self.generate_top_insights(data), 1):
            st.info(f"**{i}.** {insight}")

    def generate_coaching_summary(self, data):
        """Generate focused coaching summary."""
        summary = []
        summary.append("=" * 60)
        summary.append("GT BASEBALL COACHING SUMMARY")
        summary.append("=" * 60)
        summary.append("")

        summary.append("🥎 PITCHING INSIGHTS:")
        pv = self._numeric_series(
            data, ["PitchVelo", "pitch_velo", "pitch_velocity", "throw_velo", "velo"]
        )
        avg_velo = pv.mean()
        summary.append(
            f"• Team average velocity: {avg_velo:.1f} mph"
            if not pd.isna(avg_velo)
            else "• Team average velocity: N/A"
        )
        if "PitchOutcome" in data.columns:
            strikes = data["PitchOutcome"].isin(["Strike", "Foul"]).sum()
            strike_rate = strikes / len(data) * 100
            summary.append(
                f"• Strike rate: {strike_rate:.1f}% {'✅' if strike_rate > 65 else '⚠️'}"
            )

        pitcher_col = self._find_col(
            data, ["PitcherName", "pitcher", "pitcher_name", "player_name", "name"]
        )
        if pitcher_col and "PitchOutcome" in data.columns:
            pitcher_stats = (
                data.groupby(pitcher_col)
                .apply(
                    lambda g: pd.Series({
                        "PitchVelo_mean": self._numeric_series(
                            g, ["PitchVelo", "pitch_velo", "pitch_velocity",
                                "throw_velo", "velo"]
                        ).mean(),
                        "StrikeRate": (
                            g["PitchOutcome"].isin(["Strike", "Foul"]).sum()
                            / max(1, len(g))
                        ) * 100,
                    })
                )
                .round(1)
            )
            if not pitcher_stats.empty:
                lowest = pitcher_stats.loc[pitcher_stats["StrikeRate"].idxmin()]
                summary.append(
                    f"• Focus area: {lowest.name} — {lowest['StrikeRate']:.1f}% strike rate"
                )
        summary.append("")

        summary.append("🏏 HITTING INSIGHTS:")
        hit_data = data[data["BallInPlay"].fillna(False).astype(bool)]
        if len(hit_data) > 0 and "ExitVelo" in hit_data.columns:
            avg_exit_velo = hit_data["ExitVelo"].mean()
            hard_hits = (hit_data["ExitVelo"] > 95).sum()
            summary.append(f"• Average exit velocity: {avg_exit_velo:.1f} mph")
            summary.append(f"• Hard hit balls (>95 mph): {hard_hits}")
            if "LaunchAng" in hit_data.columns:
                optimal_la = hit_data[
                    (hit_data["LaunchAng"] >= 8) & (hit_data["LaunchAng"] <= 32)
                ]
                summary.append(
                    f"• Optimal launch angles (8-32°): {len(optimal_la)} of {len(hit_data)}"
                )
        summary.append("")

        summary.append("🛡️ DEFENSIVE INSIGHTS:")
        field_data = data[
            (data["IsEventPlayer"] == True) & (data["EventPlayerName"].notna())
        ]
        if len(field_data) > 0:
            fre = self._numeric_series(
                field_data, ["FielderRouteEfficiency", "route_efficiency", "FielderRoute"]
            )
            avg_route_eff = fre.mean()
            summary.append(
                f"• Team route efficiency: {avg_route_eff:.1f}%"
                if not pd.isna(avg_route_eff)
                else "• Team route efficiency: N/A"
            )
            if "FielderReaction" in field_data.columns:
                slow_reactions = field_data[
                    field_data["FielderReaction"]
                    > field_data["FielderReaction"].quantile(0.75)
                ]
                if len(slow_reactions) > 0:
                    summary.append(
                        f"• Players needing reaction training: {len(slow_reactions)}"
                    )

        return "\n".join(summary)

    def generate_top_insights(self, data):
        """Generate top 5 actionable insights."""
        insights = []

        pv = self._numeric_series(
            data, ["PitchVelo", "pitch_velo", "pitch_velocity", "throw_velo", "velo"]
        )
        try:
            if pv.std() > 5:
                insights.append(
                    "High velocity variance detected — work on consistent mechanics"
                )
        except Exception:
            pass

        if "PitchOutcome" in data.columns:
            strikes = data["PitchOutcome"].isin(["Strike", "Foul"]).sum()
            strike_rate = strikes / len(data) * 100
            if strike_rate < 60:
                insights.append(
                    f"Strike rate below target (currently {strike_rate:.1f}%) — "
                    "focus on command"
                )

        hit_data = data[data["BallInPlay"].fillna(False).astype(bool)]
        if len(hit_data) > 0 and "ExitVelo" in hit_data.columns:
            hard_hit_rate = (hit_data["ExitVelo"] > 95).sum() / len(hit_data) * 100
            if hard_hit_rate > 40:
                insights.append(
                    f"High hard-hit rate ({hard_hit_rate:.1f}%) — "
                    "review pitch selection/location"
                )

        field_data = data[
            (data["IsEventPlayer"] == True) & (data["EventPlayerName"].notna())
        ]
        if len(field_data) > 0 and "FielderRouteEfficiency" in field_data.columns:
            low_efficiency = field_data[field_data["FielderRouteEfficiency"] < 80]
            if len(low_efficiency) > 0:
                insights.append(
                    f"{len(low_efficiency)} fielding plays below 80% efficiency — "
                    "positioning review needed"
                )

        if "BatterTimeToFirst" in hit_data.columns:
            avg_time = hit_data["BatterTimeToFirst"].mean()
            if not pd.isna(avg_time) and avg_time > 4.5:
                insights.append(
                    f"Average time to first base: {avg_time:.2f}s — "
                    "speed training opportunity"
                )

        return insights[:5]

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def render_report_generation(self, data):
        """Render the report generation button and handle download options."""
        st.header("📋 Weekly Report Generation")

        if st.button("Generate Weekly Report"):
            report_generator = ReportGenerator(data)
            report_generator.export_csv("weekly_summary.csv")
            report_generator.export_pdf("weekly_report.pdf")
            st.success("Reports Generated! Download below.")

            for label, path, mime in [
                ("Download CSV Report", "reports/weekly_summary.csv", "text/csv"),
                ("Download PDF Report", "reports/weekly_report.pdf", "application/pdf"),
            ]:
                with open(path, "rb") as file:
                    st.download_button(
                        label=label,
                        data=file,
                        file_name=path.split("/")[-1],
                        mime=mime,
                    )

    # -----------------------------------------------------------------------
    # Video analysis prep
    # -----------------------------------------------------------------------

    def render_video_analysis_prep(self, data):
        """Prepare data for video analysis overlay."""
        st.header("🎥 Video Analysis Preparation")
        st.info("This section prepares highlight data for video overlay integration")
        st.subheader("Plays Worth Video Review")

        hard_hits = data[
            (data["ExitVelo"] > 95) & (data["BallInPlay"].fillna(False).astype(bool))
        ]
        if len(hard_hits) > 0:
            st.write("**Hard Hit Balls (>95 mph):**")
            st.dataframe(
                hard_hits[
                    ["Inning", "AtBat", "BatterName", "ExitVelo", "LaunchAng", "Result"]
                ].round(1),
                use_container_width=True,
            )

        field_data = data[
            (pd.to_numeric(data["FielderProbability"], errors="coerce").fillna(100) < 30)
            | (pd.to_numeric(data["FielderRouteEfficiency"], errors="coerce").fillna(0) > 95)
        ]
        if len(field_data) > 0:
            st.write("**Exceptional Fielding Plays:**")
            st.dataframe(
                field_data[
                    ["Inning", "AtBat", "EventPlayerName",
                     "FielderProbability", "FielderRouteEfficiency"]
                ].round(1),
                use_container_width=True,
            )

        if st.button("Export Video Analysis Data"):
            export_data = pd.concat([hard_hits, field_data]).drop_duplicates()
            st.download_button(
                label="Download Video Analysis CSV",
                data=export_data.to_csv(index=False),
                file_name="video_analysis_highlights.csv",
                mime="text/csv",
            )

    # -----------------------------------------------------------------------
    # Accountability metrics
    # -----------------------------------------------------------------------

    def render_accountability_metrics(self, data):
        """Render accountability metrics dashboard."""
        st.header("📊 Accountability Metrics")

        from accountability_analytics import AccountabilityAnalytics
        accountability = AccountabilityAnalytics(data)

        with st.expander("⚙️ Configure Standards"):
            st.write("**Baserunning Standards:**")
            col1, col2, col3 = st.columns(3)
            col1.number_input("Secondary Lead - 1B (ft)", value=16.0, key="sec_1b")
            col2.number_input("Secondary Lead - 2B (ft)", value=16.0, key="sec_2b")
            col3.number_input("Secondary Lead - 3B (ft)", value=14.0, key="sec_3b")
            st.write("**Defensive Standards:**")
            col1, col2 = st.columns(2)
            col1.number_input(
                "Route Efficiency Threshold (%)", value=85.0, key="route_threshold"
            )
            col2.number_input(
                "Reaction Time Threshold (s)", value=0.8, key="reaction_threshold"
            )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Baserunning", "Defensive Positioning", "Violations Report", "Team Summary"
        ])

        with tab1:
            st.subheader("🏃 Baserunning Accountability")
            baserunning_analysis = accountability.analyze_baserunning_accountability()
            if "error" not in baserunning_analysis and "team_summary" in baserunning_analysis:
                summary = baserunning_analysis["team_summary"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Opportunities", summary.get("total_opportunities", 0))
                col2.metric("Avg Secondary Lead", f"{summary.get('avg_secondary_lead', 0):.1f} ft")
                col3.metric("Avg Max Speed", f"{summary.get('avg_max_speed', 0):.1f} mph")
                if "compliance_metrics" in summary:
                    compliance_data = []
                    for base, metrics in summary["compliance_metrics"].items():
                        compliance_data.append({
                            "Base": base.replace("base_", "") + "B",
                            "Expected Lead (ft)": metrics["expected_lead"],
                            "Actual Avg (ft)": round(metrics["avg_lead"], 1),
                            "Compliant Plays": metrics["compliant"],
                            "Total Plays": metrics["total"],
                            "Compliance Rate": f"{metrics['compliance_rate']:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(compliance_data), use_container_width=True)
                    charts = accountability._create_baserunning_charts()
                    if "secondary_lead_comparison" in charts:
                        st.plotly_chart(
                            charts["secondary_lead_comparison"], use_container_width=True
                        )
                if (
                    "players" in baserunning_analysis
                    and baserunning_analysis["players"]
                ):
                    st.write("**Individual Player Analysis:**")
                    player = st.selectbox(
                        "Select Player", list(baserunning_analysis["players"].keys())
                    )
                    if player:
                        player_data = baserunning_analysis["players"][player]
                        col1, col2 = st.columns(2)
                        col1.metric("Total Opportunities", player_data["total_opportunities"])
                        col1.metric(
                            "Compliance Rate",
                            f"{player_data.get('compliance_rate', 0):.1f}%"
                        )
                        if "max_speed" in player_data and player_data["max_speed"]:
                            col2.metric(
                                "Avg Max Speed",
                                f"{player_data['max_speed'].get('actual_avg', 0):.1f} mph"
                            )
                            col2.metric(
                                "Max Speed Achieved",
                                f"{player_data['max_speed'].get('max_achieved', 0):.1f} mph"
                            )

        with tab2:
            st.subheader("🛡️ Defensive Positioning Accountability")
            defensive_analysis = (
                accountability.analyze_defensive_positioning_accountability()
            )
            if "error" not in defensive_analysis and "team_summary" in defensive_analysis:
                summary = defensive_analysis["team_summary"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Plays", summary.get("total_opportunities", 0))
                col2.metric(
                    "Avg Route Efficiency",
                    f"{summary.get('avg_route_efficiency', 0):.1f}%"
                )
                col3.metric(
                    "Avg Reaction Time",
                    f"{summary.get('avg_reaction_time', 0):.2f}s"
                )
                if "compliance_metrics" in summary:
                    st.write("**Team Compliance:**")
                    for metric_name, metrics in summary["compliance_metrics"].items():
                        col1, col2 = st.columns([3, 1])
                        col1.write(f"**{metric_name.replace('_', ' ').title()}:**")
                        col1.progress(metrics["compliance_rate"] / 100)
                        col2.metric("Rate", f"{metrics['compliance_rate']:.1f}%")
                if (
                    "players" in defensive_analysis
                    and defensive_analysis["players"]
                ):
                    st.write("**Individual Fielder Analysis:**")
                    fielder = st.selectbox(
                        "Select Fielder", list(defensive_analysis["players"].keys())
                    )
                    if fielder:
                        fielder_data = defensive_analysis["players"][fielder]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Plays", fielder_data["total_opportunities"])
                        col2.metric(
                            "Compliance Rate",
                            f"{fielder_data.get('compliance_rate', 0):.1f}%"
                        )
                        if "route_efficiency" in fielder_data:
                            violations = fielder_data["route_efficiency"].get(
                                "below_threshold_count", 0
                            )
                            col3.metric(
                                "Violations", violations,
                                delta=f"-{violations}" if violations > 0 else None,
                                delta_color="inverse",
                            )
                        if "route_efficiency" in fielder_data:
                            re = fielder_data["route_efficiency"]
                            st.write("**Route Efficiency:**")
                            col1, col2, col3 = st.columns(3)
                            col1.write(f"Expected: {re['expected']:.1f}%")
                            col2.write(f"Actual Avg: {re['actual_avg']:.1f}%")
                            best = re.get("best_play")
                            col3.write(
                                f"Best: {best:.1f}%"
                                if best is not None and not pd.isna(best)
                                else "Best: N/A"
                            )
                        if "reaction_time" in fielder_data:
                            rt = fielder_data["reaction_time"]
                            st.write("**Reaction Time:**")
                            col1, col2, col3 = st.columns(3)
                            col1.write(f"Expected: {rt['expected']:.2f}s")
                            col2.write(f"Actual Avg: {rt['actual_avg']:.2f}s")
                            best_r = rt.get("best_reaction")
                            col3.write(
                                f"Best: {best_r:.2f}s"
                                if best_r is not None and not pd.isna(best_r)
                                else "Best: N/A"
                            )
                    charts = accountability._create_defensive_charts()
                    if "route_efficiency_by_player" in charts:
                        st.plotly_chart(
                            charts["route_efficiency_by_player"], use_container_width=True
                        )

        with tab3:
            st.subheader("⚠️ Violations Report")
            violations_df = accountability.generate_violation_report()
            if not violations_df.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Violations", len(violations_df))
                col2.metric(
                    "High Severity",
                    len(violations_df[violations_df["Severity"] == "High"])
                    if "Severity" in violations_df else 0,
                )
                col3.metric(
                    "Medium Severity",
                    len(violations_df[violations_df["Severity"] == "Medium"])
                    if "Severity" in violations_df else 0,
                )
                col1, col2 = st.columns(2)
                with col1:
                    if "Type" in violations_df:
                        vt = st.multiselect(
                            "Type",
                            options=violations_df["Type"].unique(),
                            default=list(violations_df["Type"].unique()),
                        )
                        violations_df = violations_df[violations_df["Type"].isin(vt)]
                with col2:
                    if "Severity" in violations_df:
                        sv = st.multiselect(
                            "Severity",
                            options=violations_df["Severity"].unique(),
                            default=list(violations_df["Severity"].unique()),
                        )
                        violations_df = violations_df[violations_df["Severity"].isin(sv)]
                st.dataframe(violations_df, use_container_width=True)
                st.download_button(
                    "Download Violations Report",
                    data=violations_df.to_csv(index=False),
                    file_name=f"violations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            else:
                st.success("✅ No violations found! Team is meeting all standards.")

        with tab4:
            st.subheader("📈 Team Summary")
            br2 = accountability.analyze_baserunning_accountability()
            da2 = accountability.analyze_defensive_positioning_accountability()
            br_score = da_score = 0
            for analysis, key in [(br2, "br"), (da2, "da")]:
                if (
                    "team_summary" in analysis
                    and "compliance_metrics" in analysis["team_summary"]
                ):
                    rates = [
                        m["compliance_rate"]
                        for m in analysis["team_summary"]["compliance_metrics"].values()
                    ]
                    val = sum(rates) / len(rates) if rates else 0
                    if key == "br":
                        br_score = val
                    else:
                        da_score = val
            overall_score = (br_score + da_score) / 2
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Baserunning Compliance", f"{br_score:.1f}%",
                delta=f"{br_score - 85:.1f}%" if br_score > 0 else None,
            )
            col2.metric(
                "Defensive Compliance", f"{da_score:.1f}%",
                delta=f"{da_score - 85:.1f}%" if da_score > 0 else None,
            )
            col3.metric(
                "Overall Score", f"{overall_score:.1f}%",
                delta=f"{overall_score - 85:.1f}%" if overall_score > 0 else None,
            )
            st.write("**Detailed Breakdown:**")
            st.write("Baserunning:")
            st.progress(min(br_score / 100, 1.0))
            st.write("Defensive Positioning:")
            st.progress(min(da_score / 100, 1.0))

    # -----------------------------------------------------------------------
    # Main run
    # -----------------------------------------------------------------------

    def run_dashboard(self):
        """Main dashboard execution."""
        data = self.load_data()
        if data is None:
            return

        filtered_data, selected_pitcher, selected_batter = self.render_sidebar(data)
        self.render_overview_metrics(filtered_data)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_db = st.tabs([
            "Pitching", "Hitting", "Defensive Analytics", "Baserunning",
            "Accountability", "Game Flow", "Coaching Reports", "Video Analysis", "Database",
        ])

        with tab1:
            self.render_pitching_analysis(filtered_data)
        with tab2:
            self.render_hitting_analysis(filtered_data)
        with tab3:
            self.render_defensive_coaching_analysis(filtered_data)
        with tab4:
            self.render_baserunning_analysis(filtered_data)
        with tab5:
            self.render_accountability_metrics(filtered_data)
        with tab6:
            self.render_game_flow(filtered_data)
        with tab7:
            self.render_coaching_reports(filtered_data)
        with tab8:
            self.render_video_analysis_prep(filtered_data)
        with tab_db:
            render_database_tab()


if __name__ == "__main__":
    dashboard = GTBaseballDashboard()
    dashboard.run_dashboard()