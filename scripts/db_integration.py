"""
db_integration.py — Streamlit Database Tab for GT Baseball Dashboard
=====================================================================
Drop-in module that adds a "Database" tab to baseball_dashboard.py.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

try:
    from db_manager import GTBaseballDB
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# Singleton — reuse connection across reruns via session_state
def _get_db() -> "GTBaseballDB":
    if "gt_db" not in st.session_state:
        st.session_state.gt_db = GTBaseballDB("data/gt_baseball.db")
    return st.session_state.gt_db


# ---------------------------------------------------------------------------
# Public API — called from baseball_dashboard.py
# ---------------------------------------------------------------------------

def auto_save_to_db(df: pd.DataFrame, game_label: str, file_name: str = None):
    """
    Silently save an uploaded DataFrame to the DB.
    Called once per uploaded file inside load_data().
    Shows a small toast on success; swallows errors so the dashboard never breaks.
    """
    if not DB_AVAILABLE:
        return
    try:
        db = _get_db()
        n = db.ingest_dataframe(df, game_label=game_label,
                                file_name=file_name, skip_if_exists=True)
        if n > 0:
            st.toast(f"✅ Saved {n} rows for '{game_label}' to DB", icon="💾")
    except Exception as e:
        st.error(f"DB save failed: {type(e).__name__}: {e}")
        st.code(repr(e))
        # TEMP: also show what we tried to save
        st.write({"game_label": game_label, "file_name": file_name, "rows": len(df), "cols": list(df.columns)[:15]})


def render_database_tab():
    """Render the full Database management tab inside the dashboard."""
    st.header("🗄️ Database Management")

    if not DB_AVAILABLE:
        st.error("db_manager.py not found. Make sure it's in the same folder as baseball_dashboard.py.")
        return

    db = _get_db()
    summary = db.db_summary()

    # ── Health bar ────────────────────────────────────────────────────────
    st.subheader("Database Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Games",       summary["games"])
    col2.metric("Players",     summary["players"])
    col3.metric("Pitches",     summary["pitches"])
    col4.metric("Fielding",    summary["fielding"])
    col5.metric("Baserunning", summary["baserunning"])
    st.caption(f"📁 `{summary['db_path']}`  •  size: {summary['db_size_kb']} KB")

    st.divider()

    # ── Tabs inside the DB tab ───────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Games", "Players", "Pitching Stats", "Batting Stats",
        "Fielding Stats", "Historical Trends"
    ])

    # ── Games ─────────────────────────────────────────────────────────────
    with t1:
        st.subheader("Stored Games")
        games_df = db.list_games()

        if games_df.empty:
            st.info("No games stored yet. Upload a CSV/Parquet file to get started.")
        else:
            st.dataframe(games_df, use_container_width=True)

            # Manual upload directly into DB
            st.subheader("Add a Game from File")
            col_a, col_b = st.columns([3, 1])
            with col_a:
                new_label = st.text_input("Game label",
                    placeholder="e.g. GT vs Clemson 03/14/26")
            with col_b:
                up = st.file_uploader("CSV or Parquet", type=["csv", "parquet"],
                                      key="db_uploader")

            if st.button("💾 Save to Database") and up and new_label:
                try:
                    if up.name.lower().endswith(".parquet"):
                        import tempfile
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                        tmp.write(up.getvalue())
                        tmp.flush()
                        df_new = pd.read_parquet(tmp.name)
                        tmp.close()
                    else:
                        df_new = pd.read_csv(up)

                    n = db.ingest_dataframe(df_new, game_label=new_label,
                                            file_name=up.name, skip_if_exists=False)
                    st.success(f"✅ Saved {n} rows for '{new_label}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"DB save failed: {type(e).__name__}: {e}")
                    st.code(repr(e))
                    # TEMP: also show what we tried to save
                    st.write({"game_label": game_label, "file_name": file_name, "rows": len(df), "cols": list(df.columns)[:15]})

            # Delete a game
            st.subheader("Remove a Game")
            if not games_df.empty:
                to_delete = st.selectbox("Select game to delete",
                                         games_df["game_label"].tolist(),
                                         key="delete_select")
                if st.button("🗑️ Delete Game", type="secondary"):
                    if db.delete_game(to_delete):
                        st.success(f"Deleted '{to_delete}'")
                        st.rerun()

    # ── Players ───────────────────────────────────────────────────────────
    with t2:
        st.subheader("Player Roster")
        role_filter = st.radio("Filter by role", ["All", "pitcher", "batter", "fielder"],
                               horizontal=True)
        players_df = db.list_players(None if role_filter == "All" else role_filter)
        st.dataframe(players_df, use_container_width=True)

        # Per-player lookup
        st.subheader("Player History")
        if not players_df.empty:
            chosen = st.selectbox("Select player",
                                  players_df["player_name"].tolist(),
                                  key="player_lookup")
            role_as = st.radio("Look up as", ["any", "pitcher", "batter", "fielder"],
                               horizontal=True, key="lookup_role")
            player_df = db.query_player(chosen, role=role_as)
            if player_df.empty:
                st.info(f"No {role_as} data found for {chosen}.")
            else:
                st.dataframe(player_df, use_container_width=True)
                csv_bytes = player_df.to_csv(index=False).encode()
                st.download_button(
                    f"⬇ Download {chosen} data",
                    csv_bytes,
                    file_name=f"{chosen.replace(' ','_')}_history.csv",
                    mime="text/csv"
                )

    # ── Pitching Stats ────────────────────────────────────────────────────
    with t3:
        st.subheader("Career Pitching Stats (all games)")
        pitch_stats = db.query_pitching_stats()
        if pitch_stats.empty:
            st.info("No pitching data yet.")
        else:
            st.dataframe(pitch_stats, use_container_width=True)

            import plotly.express as px
            fig = px.bar(pitch_stats, x="PitcherName", y="AvgVelo",
                         error_y=None, title="Average Pitch Velocity by Pitcher",
                         color="TotalPitches",
                         color_continuous_scale="Blues",
                         labels={"AvgVelo": "Avg Velocity (mph)"})
            st.plotly_chart(fig, use_container_width=True)

    # ── Batting Stats ─────────────────────────────────────────────────────
    with t4:
        st.subheader("Career Batting Stats (all games)")
        bat_stats = db.query_batting_stats()
        if bat_stats.empty:
            st.info("No batting data yet.")
        else:
            st.dataframe(bat_stats, use_container_width=True)

            import plotly.express as px
            fig = px.scatter(
                bat_stats.dropna(subset=["AvgExitVelo", "AvgLaunchAngle"]),
                x="AvgLaunchAngle", y="AvgExitVelo",
                text="BatterName", size="PlateAppearances",
                title="Exit Velocity vs Launch Angle (career avg)",
                labels={"AvgExitVelo": "Avg Exit Velo (mph)",
                        "AvgLaunchAngle": "Avg Launch Angle (°)"}
            )
            fig.add_hrect(y0=95, y1=bat_stats["AvgExitVelo"].max() + 5,
                          fillcolor="green", opacity=0.1,
                          annotation_text="Hard Hit Zone")
            fig.add_vrect(x0=8, x1=32,
                          fillcolor="red", opacity=0.1,
                          annotation_text="Barrel Zone")
            st.plotly_chart(fig, use_container_width=True)

    # ── Fielding Stats ────────────────────────────────────────────────────
    with t5:
        st.subheader("Career Fielding Stats (all games)")
        field_stats = db.query_fielding_stats()
        if field_stats.empty:
            st.info("No fielding data yet.")
        else:
            st.dataframe(field_stats, use_container_width=True)

            import plotly.express as px
            plot_df = field_stats.dropna(subset=["AvgRouteEfficiency", "AvgReactionTime"])
            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="AvgReactionTime", y="AvgRouteEfficiency",
                    text="FielderName", size="TotalPlays",
                    title="Route Efficiency vs Reaction Time",
                    labels={"AvgReactionTime": "Avg Reaction Time (s)",
                            "AvgRouteEfficiency": "Avg Route Efficiency (%)"}
                )
                fig.add_hline(y=85, line_dash="dash", line_color="green",
                              annotation_text="85% target")
                st.plotly_chart(fig, use_container_width=True)

    # ── Historical Trends ─────────────────────────────────────────────────
    with t6:
        st.subheader("Game-by-Game Historical Trends")
        trends = db.query_historical_trends()
        if len(trends) < 2:
            st.info("Load at least 2 games to see historical trends.")
            if not trends.empty:
                st.dataframe(trends, use_container_width=True)
        else:
            st.dataframe(trends, use_container_width=True)

            import plotly.express as px

            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(trends, x="game_label", y="AvgPitchVelo",
                              markers=True, title="Avg Pitch Velocity Game-by-Game",
                              labels={"AvgPitchVelo": "mph",
                                      "game_label": "Game"})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.line(trends, x="game_label", y="AvgExitVelo",
                              markers=True, title="Avg Exit Velocity Game-by-Game",
                              labels={"AvgExitVelo": "mph",
                                      "game_label": "Game"})
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                fig = px.line(trends, x="game_label", y="AvgRouteEfficiency",
                              markers=True, title="Avg Route Efficiency Game-by-Game",
                              labels={"AvgRouteEfficiency": "%",
                                      "game_label": "Game"})
                fig.add_hline(y=85, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                fig = px.line(trends, x="game_label", y="AvgRunnerSpeed",
                              markers=True, title="Avg Runner Speed Game-by-Game",
                              labels={"AvgRunnerSpeed": "mph",
                                      "game_label": "Game"})
                st.plotly_chart(fig, use_container_width=True)

            # Full export
            csv_bytes = trends.to_csv(index=False).encode()
            st.download_button("⬇ Download Trends CSV", csv_bytes,
                               file_name="historical_trends.csv",
                               mime="text/csv")