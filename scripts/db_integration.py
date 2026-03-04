"""
db_integration.py — Streamlit Database Tab for GT Baseball Dashboard
=====================================================================
Drop-in module that adds a "Database" tab to baseball_dashboard.py.

Updated to handle new dict return values from db_manager:
  {"status": "inserted",  "rows": N}
  {"status": "skipped",   "rows": 0}
  {"status": "duplicate", "rows": 0, "matching_game": "<label>"}

Duplicate handling in auto_save_to_db():
  - On duplicate detected: stores warning in st.session_state and shows
    a Streamlit warning with a "Save Anyway" button in the sidebar.
  - User clicks "Save Anyway" -> re-calls ingest with force=True.
"""

import os
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px  # type: ignore
from pathlib import Path

try:
    from db_manager import GTBaseballDB
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Singleton DB — reuse across reruns via session_state
# ---------------------------------------------------------------------------

def _get_db() -> "GTBaseballDB":
    if "gt_db" not in st.session_state:
        st.session_state.gt_db = GTBaseballDB("data/gt_baseball.db")
    return st.session_state.gt_db


def _invalidate_cache():
    """Bump the cache version so every cached query re-fetches on next access."""
    st.session_state["_db_cache_version"] = st.session_state.get("_db_cache_version", 0) + 1


def _cached_db_query(cache_key: str, fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs), cache the result in session_state[cache_key].
    Re-fetches automatically whenever _invalidate_cache() has been called.
    """
    current_ver = st.session_state.get("_db_cache_version", 0)
    cached = st.session_state.get(cache_key)
    if cached is None or cached["version"] != current_ver:
        st.session_state[cache_key] = {"version": current_ver, "data": fn(*args, **kwargs)}
    return st.session_state[cache_key]["data"]


# ---------------------------------------------------------------------------
# Public API — called from baseball_dashboard.py
# ---------------------------------------------------------------------------

def auto_save_to_db(df: pd.DataFrame, game_label: str, file_name: str = None):
    """
    Silently save an uploaded DataFrame to the DB after a file upload.
    Called once per file inside load_data().

    Handles all three return statuses:
      inserted  → small success toast
      skipped   → silent (already saved)
      duplicate → warning banner + "Save Anyway" button stored in session_state
    """
    if not DB_AVAILABLE:
        return
    try:
        db = _get_db()
        result = db.ingest_dataframe(
            df, game_label=game_label, file_name=file_name, skip_if_exists=True
        )

        if result["status"] == "inserted":
            _invalidate_cache()
            st.toast(f"✅ Saved {result['rows']} rows for '{game_label}'", icon="💾")

        elif result["status"] == "skipped":
            pass  # already in DB, no noise needed

        elif result["status"] == "duplicate":
            # Store in session_state so the sidebar can render the warning
            st.session_state["_db_duplicate_pending"] = {
                "df":            df,
                "game_label":    game_label,
                "file_name":     file_name,
                "matching_game": result["matching_game"],
            }

    except Exception as e:
        st.toast(f"⚠️ DB save failed: {e}", icon="⚠️")


def render_duplicate_warning():
    """
    Call this once near the top of your sidebar to render a pending
    duplicate warning + "Save Anyway" button if one exists.
    Clears itself after the user acts.
    """
    if not DB_AVAILABLE:
        return
    pending = st.session_state.get("_db_duplicate_pending")
    if not pending:
        return

    st.sidebar.warning(
        f"⚠️ **Possible duplicate detected**\n\n"
        f"The file you uploaded looks very similar to an existing game:\n\n"
        f"**Matching game:** `{pending['matching_game']}`\n\n"
        f"This could mean you uploaded the same file twice, or it could be "
        f"a different half of the same game. Check before saving."
    )

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("💾 Save Anyway", key="_dup_force_save"):
            try:
                db = _get_db()
                result = db.ingest_dataframe(
                    pending["df"],
                    game_label=pending["game_label"],
                    file_name=pending["file_name"],
                    skip_if_exists=False,
                    force=True,
                )
                _invalidate_cache()
                st.toast(
                    f"✅ Force-saved {result['rows']} rows for '{pending['game_label']}'",
                    icon="💾"
                )
            except Exception as e:
                st.toast(f"⚠️ Force save failed: {e}", icon="⚠️")
            finally:
                del st.session_state["_db_duplicate_pending"]
                st.rerun()

    with col2:
        if st.button("✖ Discard", key="_dup_discard"):
            del st.session_state["_db_duplicate_pending"]
            st.rerun()


# ---------------------------------------------------------------------------
# Full Database Tab UI
# ---------------------------------------------------------------------------

def render_database_tab():
    """Render the full Database management tab inside the dashboard."""
    st.header("🗄️ Database Management")
    if st.button("🔄 Refresh Data", help="Re-fetch all stats from the database"):
        _invalidate_cache()
        st.rerun()

    if not DB_AVAILABLE:
        st.error(
            "db_manager.py not found. "
            "Make sure it is in the same folder as baseball_dashboard.py."
        )
        return

    db = _get_db()
    summary = _cached_db_query("_db_summary", db.db_summary)

    # ── Health metrics ─────────────────────────────────────────────────
    st.subheader("Database Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games",       summary["games"])
    c2.metric("GT Players",  summary["players"])
    c3.metric("Pitches",     summary["pitches"])
    c4.metric("Fielding",    summary["fielding"])
    c5.metric("Baserunning", summary["baserunning"])
    if "db_location" in summary:
        st.caption(f"☁️ Turso: `{summary['db_location']}`")
    else:
        st.caption(f"📁 `{summary['db_path']}`  •  {summary['db_size_kb']} KB")

    st.divider()

    # ── Sub-tabs ───────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Games", "GT Roster", "Pitching Stats", "Batting Stats",
        "Fielding Stats", "Historical Trends"
    ])

    # ── Games ─────────────────────────────────────────────────────────
    with t1:
        st.subheader("Stored Games")
        games_df = _cached_db_query("_db_games", db.list_games)

        if games_df.empty:
            st.info("No games stored yet. Upload a CSV/Parquet file to get started.")
        else:
            st.dataframe(games_df, use_container_width=True)

        # Manual file upload directly into DB
        st.subheader("Add a Game Directly")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            new_label = st.text_input(
                "Game label", placeholder="e.g. GT vs Clemson 03/14/26"
            )
        with col_b:
            up = st.file_uploader(
                "CSV or Parquet", type=["csv", "parquet"], key="db_uploader"
            )

        if st.button("💾 Save to Database") and up and new_label:
            try:
                if up.name.lower().endswith(".parquet"):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                    tmp.write(up.getvalue())
                    tmp.flush()
                    df_new = pd.read_parquet(tmp.name)
                    tmp.close()
                    os.unlink(tmp.name)
                else:
                    df_new = pd.read_csv(up)

                result = db.ingest_dataframe(
                    df_new, game_label=new_label,
                    file_name=up.name, skip_if_exists=False
                )

                if result["status"] == "inserted":
                    _invalidate_cache()
                    st.success(f"✅ Saved {result['rows']} rows for '{new_label}'")
                    st.rerun()
                elif result["status"] == "duplicate":
                    st.warning(
                        f"⚠️ This file looks identical to **{result['matching_game']}**. "
                        "Use the checkbox below to force-save anyway."
                    )
                    if st.checkbox("Yes, save it anyway", key="force_manual"):
                        result2 = db.ingest_dataframe(
                            df_new, game_label=new_label,
                            file_name=up.name, skip_if_exists=False, force=True
                        )
                        _invalidate_cache()
                        st.success(f"✅ Force-saved {result2['rows']} rows for '{new_label}'")
                        st.rerun()

            except Exception as e:
                st.error(f"Failed: {e}")

        # Delete a game
        if not games_df.empty:
            st.subheader("Remove a Game")
            to_delete = st.selectbox(
                "Select game to delete",
                games_df["game_label"].tolist(),
                key="delete_select"
            )
            if st.button("🗑️ Delete Game", type="secondary"):
                if db.delete_game(to_delete):
                    _invalidate_cache()
                    st.success(f"Deleted '{to_delete}'")
                    st.rerun()

    # ── GT Roster ─────────────────────────────────────────────────────
    with t2:
        st.subheader("GT Player Roster")
        st.caption(
            "Only players in this table have named records linked to their stats. "
            "Opponent players appear in the data tables with NULL player IDs."
        )

        role_filter = st.radio(
            "Filter by role",
            ["All", "pitcher", "batter", "fielder", "multiple"],
            horizontal=True
        )
        role_key = role_filter.lower()
        players_df = _cached_db_query(f"_db_players_{role_key}", db.list_players, None if role_filter == "All" else role_filter)

        if players_df.empty:
            st.info(
                "No GT players found. Make sure `scripts/gt_roster.csv` exists "
                "and re-upload your game files."
            )
        else:
            st.dataframe(players_df, use_container_width=True)

        # Per-player history lookup
        st.subheader("Player History")
        if not players_df.empty:
            chosen = st.selectbox(
                "Select player",
                players_df["player_name"].tolist(),
                key="player_lookup"
            )
            role_as = st.radio(
                "Look up as", ["any", "pitcher", "batter", "fielder"],
                horizontal=True, key="lookup_role"
            )
            player_df = _cached_db_query(f"_db_player_{chosen}_{role_as}", db.query_player, chosen, role=role_as)
            if player_df.empty:
                st.info(f"No {role_as} data found for {chosen}.")
            else:
                st.dataframe(player_df, use_container_width=True)
                st.download_button(
                    f"⬇ Download {chosen} data",
                    player_df.to_csv(index=False).encode(),
                    file_name=f"{chosen.replace(' ','_')}_history.csv",
                    mime="text/csv"
                )

        # Reload roster button
        st.divider()
        if st.button("🔄 Reload Roster from CSV"):
            db.reload_roster()
            _invalidate_cache()
            st.success("Roster reloaded.")
            st.rerun()

    # ── Pitching Stats ────────────────────────────────────────────────
    with t3:
        st.subheader("Career Pitching Stats — GT Pitchers")
        pitch_stats = _cached_db_query("_db_pitch_stats", db.query_pitching_stats)
        if pitch_stats.empty:
            st.info("No GT pitching data yet.")
        else:
            st.dataframe(pitch_stats, use_container_width=True)
            fig = px.bar(
                pitch_stats, x="PitcherName", y="AvgVelo",
                title="Average Pitch Velocity by Pitcher",
                color="TotalPitches", color_continuous_scale="Blues",
                labels={"AvgVelo": "Avg Velocity (mph)"}
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Batting Stats ─────────────────────────────────────────────────
    with t4:
        st.subheader("Career Batting Stats — GT Batters")
        bat_stats = _cached_db_query("_db_bat_stats", db.query_batting_stats)
        if bat_stats.empty:
            st.info("No GT batting data yet.")
        else:
            st.dataframe(bat_stats, use_container_width=True)
            plot_df = bat_stats.dropna(subset=["AvgExitVelo", "AvgLaunchAngle"])
            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="AvgLaunchAngle", y="AvgExitVelo",
                    text="BatterName", size="PlateAppearances",
                    title="Exit Velocity vs Launch Angle (career avg)",
                    labels={"AvgExitVelo": "Avg Exit Velo (mph)",
                            "AvgLaunchAngle": "Avg Launch Angle (°)"}
                )
                fig.add_hrect(
                    y0=95, y1=bat_stats["AvgExitVelo"].max() + 5,
                    fillcolor="green", opacity=0.1, annotation_text="Hard Hit Zone"
                )
                fig.add_vrect(
                    x0=8, x1=32,
                    fillcolor="red", opacity=0.1, annotation_text="Barrel Zone"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Fielding Stats ────────────────────────────────────────────────
    with t5:
        st.subheader("Career Fielding Stats — GT Fielders")
        field_stats = _cached_db_query("_db_field_stats", db.query_fielding_stats)
        if field_stats.empty:
            st.info("No GT fielding data yet.")
        else:
            st.dataframe(field_stats, use_container_width=True)
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
                fig.add_hline(
                    y=85, line_dash="dash", line_color="green",
                    annotation_text="85% target"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Historical Trends ─────────────────────────────────────────────
    with t6:
        st.subheader("Game-by-Game Historical Trends")
        trends = _cached_db_query("_db_trends", db.query_historical_trends)
        if len(trends) < 2:
            st.info("Load at least 2 games to see historical trends.")
            if not trends.empty:
                st.dataframe(trends, use_container_width=True)
        else:
            st.dataframe(trends, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(trends, x="game_label", y="AvgPitchVelo", markers=True,
                              title="Avg Pitch Velocity Game-by-Game",
                              labels={"AvgPitchVelo": "mph", "game_label": "Game"})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.line(trends, x="game_label", y="AvgExitVelo", markers=True,
                              title="Avg Exit Velocity Game-by-Game",
                              labels={"AvgExitVelo": "mph", "game_label": "Game"})
                st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig = px.line(trends, x="game_label", y="AvgRouteEfficiency", markers=True,
                              title="Avg Route Efficiency Game-by-Game",
                              labels={"AvgRouteEfficiency": "%", "game_label": "Game"})
                fig.add_hline(y=85, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
            with c4:
                fig = px.line(trends, x="game_label", y="AvgRunnerSpeed", markers=True,
                              title="Avg Runner Speed Game-by-Game",
                              labels={"AvgRunnerSpeed": "mph", "game_label": "Game"})
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇ Download Trends CSV",
                trends.to_csv(index=False).encode(),
                file_name="historical_trends.csv",
                mime="text/csv"
            )