import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from data_loader import GTBaseballDataLoader
from baseball_analyzer import GTBaseballAnalyzer
from report_generator import ReportGenerator
from db_integration import render_database_tab, auto_save_to_db


class GTBaseballDashboard:
    """
    Interactive Streamlit dashboard for GT Baseball 6th Tool analytics.
    """
    
    def __init__(self):
        self.setup_page_config()
        
    # --- Helpers to tolerate different column names / missing columns ---
    def _find_col(self, df, candidates):
        """Return first existing column name from candidates, or None."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _numeric_series(self, df, candidates):
        """Return numeric Series for the first matching candidate, coerced to numeric; else NaN series."""
        col = self._find_col(df, candidates)
        if col is None:
            return pd.Series([pd.NA] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    def _name_series(self, df, candidates):
        """Return object Series for the first matching candidate, else Series of empty strings."""
        col = self._find_col(df, candidates)
        if col is None:
            return pd.Series([""] * len(df), index=df.index)
        return df[col].astype(str).fillna("")

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="GT Baseball 6th Tool Dashboard",
            page_icon="⚾",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #B3A369;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 3px solid #003057;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #B3A369;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_data(self):
        """Load and cache data."""
        if 'game_data' not in st.session_state:
            try:
                from db_integration import _get_db
                db = _get_db()
                summary = db.db_summary()
                if summary["games"] > 0:
                    st.info(f"📦 Database has {summary['games']} games and {summary['pitches']} pitches stored.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load All Games from Database"):
                            st.session_state.game_data = db.query_all_games()
                            st.rerun()
                    with col2:
                        games_df = db.list_games()
                        chosen = st.selectbox("Or load one game:", 
                                            ["— pick one —"] + games_df["game_label"].tolist())
                        if chosen != "— pick one —":
                            st.session_state.game_data = db.query_game(chosen)
                            st.rerun()
                    st.divider()
            except Exception:
                st.warning("⚠️ Database not available or inaccessible.")
            loader = GTBaseballDataLoader()
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload GT Baseball CSV/Parquet files",
                accept_multiple_files=True,
                type=['csv', 'parquet']
            )
            
            if uploaded_files:
                all_data = []
                for i, file in enumerate(uploaded_files):
                    try:
                        # handle parquet or csv uploads
                        if file.name.lower().endswith(".parquet"):
                            import tempfile
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                            tmp.write(file.getvalue())
                            tmp.flush()
                            df = pd.read_parquet(tmp.name)
                            tmp.close()
                        else:
                            df = pd.read_csv(file)

                        # Normalize this file's columns to canonical names right away
                        df = loader._normalize_columns(df)

                        df['GameID'] = f"Game_{i+1}"
                        df['FileName'] = file.name
                        all_data.append(df)
                        auto_save_to_db(df, game_label=file.name, file_name=file.name)
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")
                
                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)

                    # (combined_data already contains canonical columns from per-file normalization)
                    # Apply the same cleaning/derived steps
                    loader.data = combined_data
                    st.session_state.game_data = loader._clean_data(combined_data)
                    st.session_state.game_data = loader._add_derived_columns(st.session_state.game_data)

                    # Run additional inference to fill BallInPlay / EventPlayerName where possible
                    try:
                        st.session_state.game_data = loader._infer_bip_and_fielding(st.session_state.game_data)
                    except Exception:
                        # don't fail UI on inference issues
                        pass

                    # Diagnostics: show counts and give helpful collapsible info if no BIP/fielding rows
                    bip_count = 0
                    try:
                        bip_count = int(st.session_state.game_data["BallInPlay"].astype(bool).sum())
                    except Exception:
                        bip_count = 0
                    fielding_rows = 0
                    try:
                        fielding_rows = int(len(st.session_state.game_data[(st.session_state.game_data.get("IsEventPlayer")==True) & (st.session_state.game_data.get("EventPlayerName").notna())]))
                    except Exception:
                        fielding_rows = 0

                    if bip_count == 0:
                        st.warning("No balls in play detected after automatic inference. Hitting charts will be empty.")
                        with st.expander("Why no BIP? See columns & sample rows"):
                            st.write("Available columns (first 120):")
                            st.write(st.session_state.game_data.columns.tolist()[:120])
                            st.write("Sample rows:")
                            st.dataframe(st.session_state.game_data.head(5), use_container_width=True)

                    if fielding_rows == 0:
                        st.info("No fielding/event-player rows detected. Defensive analytics will be limited.")
                        with st.expander("Why no fielding data? See candidate columns"):
                            st.write("Look for these columns in your file: 'primary_fielder', 'player_involved', 'lead_baserunner', 'route_efficiency', 'reaction_time', 'Probability'")
                            st.write("Available columns (first 120):")
                            st.write(st.session_state.game_data.columns.tolist()[:120])
                            st.write("Sample rows:")
                            st.dataframe(st.session_state.game_data.head(5), use_container_width=True)

                    st.success(f"Loaded {len(all_data)} games with {len(st.session_state.game_data)} total pitches")
                    return st.session_state.game_data
            else:
                st.info("Please upload CSV files to begin analysis")
                return None
        
        return st.session_state.game_data
    
    def render_sidebar(self, data):
        """Render sidebar with filters and controls."""
        st.sidebar.header("🎛️ Analysis Controls")
        
        # Game selection
        if 'GameID' in data.columns:
            games = ['All Games'] + list(data['GameID'].unique())
            selected_game = st.sidebar.selectbox("Select Game:", games)
            
            if selected_game != 'All Games':
                data = data[data['GameID'] == selected_game]
        
        # Player filters
        st.sidebar.subheader("Player Filters")
        
        # find best available column for pitcher / batter (tolerate many exports)
        pitcher_col = None
        for c in ['PitcherName', 'pitcher', 'pitcher_name', 'pitcher_id', 'player_name', 'name']:
            if c in data.columns:
                pitcher_col = c
                break

        batter_col = None
        for c in ['BatterName', 'batter', 'batter_name', 'batter_id', 'player_name', 'name']:
            if c in data.columns:
                batter_col = c
                break

        pitchers = ['All Pitchers']
        if pitcher_col:
            pitchers += list(data[pitcher_col].dropna().unique())
        batters = ['All Batters']
        if batter_col:
            batters += list(data[batter_col].dropna().unique())

        selected_pitcher = st.selectbox("Pitcher", pitchers)
        selected_batter = st.selectbox("Batter", batters)

        # apply filters using the detected columns (if present)
        filtered = data.copy()
        if selected_pitcher != 'All Pitchers' and pitcher_col:
            filtered = filtered[filtered[pitcher_col] == selected_pitcher]
        if selected_batter != 'All Batters' and batter_col:
            filtered = filtered[filtered[batter_col] == selected_batter]
        return filtered, selected_pitcher, selected_batter
    
    def render_overview_metrics(self, data):
        """Render key performance metrics."""
        st.markdown('<div class="main-header">GT Baseball 6th Tool Analytics</div>', unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_pitches = len(data)
            st.metric("Total Pitches", f"{total_pitches:,}")
        
        with col2:
            pv = self._numeric_series(data, ['PitchVelo', 'pitch_velo', 'pitch_velocity', 'throw_velo', 'velo'])
            avg_velocity = pv.mean()
            if pd.isna(avg_velocity):
                st.metric("Avg Pitch Velocity", "N/A")
            else:
                st.metric("Avg Pitch Velocity", f"{avg_velocity:.1f} mph")
        
        with col3:
            bip = self._numeric_series(data, ['BallInPlay', 'ball_in_play', 'BallInPlay'])
            # bip may be bool-like; coerce to boolean then sum
            try:
                bip_count = int(bip.astype(bool).sum())
            except Exception:
                bip_count = int(bip.dropna().sum()) if bip.dropna().size > 0 else 0
            st.metric("Balls in Play", f"{bip_count:,}")
        
        with col4:
            ev = self._numeric_series(data, ['ExitVelo', 'exit_velo', 'ExitVelocity', 'Exit_Velo'])
            bip_mask = self._numeric_series(data, ['BallInPlay', 'ball_in_play', 'BallInPlay']).fillna(0).astype(bool)
            if ev.dropna().size > 0 and bip_mask.any():
                avg_exit_velo = ev[bip_mask].mean()
                if pd.isna(avg_exit_velo):
                    st.metric("Avg Exit Velocity", "N/A")
                else:
                    st.metric("Avg Exit Velocity", f"{avg_exit_velo:.1f} mph")
            else:
                st.metric("Avg Exit Velocity", "N/A")
        
        with col5:
            pser = self._name_series(data, ['PitcherName', 'pitcher', 'pitcher_name', 'player_name', 'name'])
            bser = self._name_series(data, ['BatterName', 'batter', 'batter_name', 'player_name', 'name'])
            unique_players = len(set(pser.dropna().unique()) | set(bser.dropna().unique()))
            st.metric("Total Players", unique_players)
    
    def render_pitching_analysis(self, data):
        """Render pitching analysis section."""
        st.header("🥎 Pitching Analysis")
        
        # prepare a numeric pitch velocity series from best available column names
        pv = self._numeric_series(data, ['PitchVelo', 'pitch_velo', 'pitch_velocity', 'throw_velo', 'velo'])
        temp = data.copy()
        temp["_PitchVelo"] = pv
        
        # If no velocity data, show friendly message and skip velocity charts
        if temp["_PitchVelo"].dropna().empty:
            st.warning("No pitch velocity data available for pitching analysis.")
            # still show pitch outcome pie if present
            if 'PitchOutcome' in data.columns:
                outcome_counts = data['PitchOutcome'].value_counts()
                fig_outcomes = px.pie(
                    values=outcome_counts.values,
                    names=outcome_counts.index,
                    title="Pitch Outcome Distribution"
                )
                st.plotly_chart(fig_outcomes, use_container_width=True)
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Velocity distribution (use normalized temp column)
            fig_vel = px.histogram(
                temp.dropna(subset=['_PitchVelo']),
                x='_PitchVelo',
                nbins=20,
                title="Pitch Velocity Distribution",
                labels={'_PitchVelo': 'Velocity (mph)', 'count': 'Frequency'}
            )
            mean_vel = temp["_PitchVelo"].mean()
            fig_vel.add_vline(x=mean_vel, line_dash="dash", 
                              annotation_text=f"Mean: {mean_vel:.1f}")
            st.plotly_chart(fig_vel, use_container_width=True)
        
        with col2:
            # Pitch outcomes (if present)
            if 'PitchOutcome' in data.columns:
                outcome_counts = data['PitchOutcome'].value_counts()
                fig_outcomes = px.pie(
                    values=outcome_counts.values,
                    names=outcome_counts.index,
                    title="Pitch Outcome Distribution"
                )
                st.plotly_chart(fig_outcomes, use_container_width=True)
        
        # Velocity by inning (use temp _PitchVelo and guard missing Inning)
        if 'Inning' in temp.columns and temp['Inning'].nunique() > 1:
            st.subheader("Velocity by Inning")
            df_box = temp.dropna(subset=['_PitchVelo', 'Inning'])
            if len(df_box) > 0:
                fig_inning = px.box(
                    df_box,
                    x='Inning',
                    y='_PitchVelo',
                    title="Pitch Velocity by Inning",
                    labels={'_PitchVelo': 'Velocity (mph)'}
                )
                st.plotly_chart(fig_inning, use_container_width=True)
        
        # Pitcher comparison: build stats using detected pitcher column and normalized velocity
        st.subheader("Pitcher Comparison")
        pitcher_col = self._find_col(temp, ['PitcherName', 'pitcher', 'pitcher_name', 'player_name', 'name'])
        if pitcher_col:
            # average velocity and pitch counts
            pv_by_pitcher = temp.dropna(subset=[pitcher_col]).groupby(pitcher_col)["_PitchVelo"].agg(['mean', 'count'])
            pv_by_pitcher = pv_by_pitcher.rename(columns={'mean': 'Avg Velocity', 'count': 'Total Pitches'})
            # strike rate if PitchOutcome exists
            if 'PitchOutcome' in temp.columns:
                strike_rate = temp.groupby(pitcher_col)['PitchOutcome'].apply(lambda s: (s.isin(['Strike', 'Foul']).sum() / max(1, len(s))) * 100)
                pv_by_pitcher['Strike Rate %'] = strike_rate
            else:
                pv_by_pitcher['Strike Rate %'] = pd.NA
            pitcher_stats = pv_by_pitcher.round(2)
            st.dataframe(pitcher_stats, use_container_width=True)
        else:
            st.info("No pitcher identifier column found for pitcher comparison.")
    
    def render_hitting_analysis(self, data):
        """Render hitting analysis section."""
        st.header("🏏 Hitting Analysis")
        
        hit_data = data[data['BallInPlay'].fillna(False).astype(bool)]

        
        if len(hit_data) == 0:
            st.warning("No balls in play data available for hitting analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Exit velocity distribution
            if 'ExitVelo' in hit_data.columns:
                fig_exit = px.histogram(
                    hit_data.dropna(subset=['ExitVelo']),
                    x='ExitVelo',
                    nbins=15,
                    title="Exit Velocity Distribution",
                    labels={'ExitVelo': 'Exit Velocity (mph)'}
                )
                fig_exit.add_vline(x=95, line_dash="dash", line_color="red",
                                  annotation_text="Hard Hit Threshold")
                st.plotly_chart(fig_exit, use_container_width=True)
        
        with col2:
            # Launch angle distribution
            if 'LaunchAng' in hit_data.columns:
                fig_launch = px.histogram(
                    hit_data.dropna(subset=['LaunchAng']),
                    x='LaunchAng',
                    nbins=15,
                    title="Launch Angle Distribution",
                    labels={'LaunchAng': 'Launch Angle (degrees)'}
                )
                st.plotly_chart(fig_launch, use_container_width=True)
        
        # Exit velocity vs Launch angle scatter
        if 'ExitVelo' in hit_data.columns and 'LaunchAng' in hit_data.columns:
            st.subheader("Exit Velocity vs Launch Angle")
            
            scatter_data = hit_data.dropna(subset=['ExitVelo', 'LaunchAng'])
            if len(scatter_data) > 0:
                fig_scatter = px.scatter(
                    scatter_data,
                    x='LaunchAng',
                    y='ExitVelo',
                    color='Result',
                    title="Exit Velocity vs Launch Angle",
                    labels={'LaunchAng': 'Launch Angle (degrees)', 'ExitVelo': 'Exit Velocity (mph)'}
                )
                
                # Add quality zones
                fig_scatter.add_hrect(y0=95, y1=scatter_data['ExitVelo'].max(), 
                                     fillcolor="green", opacity=0.2,
                                     annotation_text="Hard Hit Zone")
                fig_scatter.add_vrect(x0=8, x1=32, fillcolor="red", opacity=0.2,
                                     annotation_text="Barrel Zone")
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Sprint speed analysis
        if 'BatterTimeToFirst' in hit_data.columns:
            sprint_data = hit_data.dropna(subset=['BatterTimeToFirst'])
            if len(sprint_data) > 0:
                st.subheader("Sprint Speed Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_sprint = px.histogram(
                        sprint_data,
                        x='BatterTimeToFirst',
                        title="Time to First Base Distribution",
                        labels={'BatterTimeToFirst': 'Time to First (seconds)'}
                    )
                    st.plotly_chart(fig_sprint, use_container_width=True)
                
                with col2:
                    # Top sprint speeds
                    fastest_runners = sprint_data.nsmallest(10, 'BatterTimeToFirst')[['BatterName', 'BatterTimeToFirst']]
                    st.subheader("Fastest Times to First")
                    st.dataframe(fastest_runners, use_container_width=True)
    
    def render_fielding_analysis(self, data):
        """Render fielding analysis section."""
        st.header("🥅 Fielding Analysis")
        
        field_data = data[(data['IsEventPlayer'] == True) & (data['EventPlayerName'].notna())]
        
        if len(field_data) == 0:
            st.warning("No fielding data available for analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Route efficiency
            if 'FielderRouteEfficiency' in field_data.columns:
                fig_route = px.histogram(
                    field_data.dropna(subset=['FielderRouteEfficiency']),
                    x='FielderRouteEfficiency',
                    title="Route Efficiency Distribution",
                    labels={'FielderRouteEfficiency': 'Route Efficiency (%)'}
                )
                st.plotly_chart(fig_route, use_container_width=True)
        
        with col2:
            # Reaction time
            if 'FielderReaction' in field_data.columns:
                fig_reaction = px.histogram(
                    field_data.dropna(subset=['FielderReaction']),
                    x='FielderReaction',
                    title="Reaction Time Distribution",
                    labels={'FielderReaction': 'Reaction Time (seconds)'}
                )
                st.plotly_chart(fig_reaction, use_container_width=True)
        
        # Fielder performance comparison
        if 'FielderRouteEfficiency' in field_data.columns and 'FielderReaction' in field_data.columns:
            st.subheader("Fielder Performance Comparison")
            
            fielder_stats = field_data.groupby('EventPlayerName').agg({
                'FielderRouteEfficiency': 'mean',
                'FielderReaction': 'mean',
                'FielderMaxSpeed': 'max'
            }).round(2)
            
            fielder_stats.columns = ['Avg Route Efficiency', 'Avg Reaction Time', 'Max Speed']
            st.dataframe(fielder_stats, use_container_width=True)
        
        # Catch probability analysis
        if 'FielderProbability' in field_data.columns:
            prob_data = field_data.dropna(subset=['FielderProbability'])
            if len(prob_data) > 0:
                st.subheader("Catch Probability Analysis")
                
                fig_prob = px.scatter(
                    prob_data,
                    x='FielderProbability',
                    y='FielderRouteEfficiency',
                    color='EventPlayerName',
                    title="Route Efficiency vs Catch Probability",
                    labels={'FielderProbability': 'Catch Probability (%)', 
                           'FielderRouteEfficiency': 'Route Efficiency (%)'}
                )
                st.plotly_chart(fig_prob, use_container_width=True)
    
    def render_baserunning_analysis(self, data):
        """Render baserunning analysis section."""
        st.header("🏃 Baserunning Analysis")
        
        # Check if baserunning columns exist using the helper method
        speed_col = self._find_col(data, [
            'BaserunnerMaxSpeed', 'baserunner_max_speed', 'max_speed', 
            'MaxSpeed', 'runner_speed', 'speed'
        ])
        
        if speed_col is None:
            st.warning("No baserunning speed data available for analysis.")
            return
        
        base_data = data[data[speed_col].notna()]
        
        if len(base_data) == 0:
            st.warning("No baserunning data available for analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Max speed distribution
            fig_speed = px.histogram(
                base_data,
                x=speed_col,
                title="Baserunner Max Speed Distribution",
                labels={speed_col: 'Max Speed (mph)'}
            )
            st.plotly_chart(fig_speed, use_container_width=True)
        
        with col2:
            # Speed by base position - check if these columns exist
            initial_col = self._find_col(base_data, [
                'BaserunnerInitial', 'baserunner_initial', 'base_start', 
                'initial_base', 'from_base'
            ])
            final_col = self._find_col(base_data, [
                'BaserunnerFinal', 'baserunner_final', 'base_end', 
                'final_base', 'to_base'
            ])
            
            if initial_col and final_col:
                base_data_clean = base_data.dropna(subset=[initial_col, final_col])
                if len(base_data_clean) > 0:
                    fig_bases = px.box(
                        base_data_clean,
                        x=initial_col,
                        y=speed_col,
                        title="Speed by Starting Base",
                        labels={initial_col: 'Starting Base', speed_col: 'Max Speed (mph)'}
                    )
                    st.plotly_chart(fig_bases, use_container_width=True)
                else:
                    st.info("Insufficient data for base-by-base speed analysis.")
            else:
                st.info("Base position data not available for detailed analysis.")
    
    def render_game_flow(self, data):
        """Render game flow analysis."""
        st.header("📊 Game Flow Analysis")
        
        if 'Inning' not in data.columns:
            st.warning("No inning data available for game flow analysis.")
            return
        
        # Pitches per inning
        inning_summary = data.groupby('Inning').agg({
            'PitchVelo': 'mean',
            'AtBat': 'nunique',
            'PitchOutcome': 'count'
        }).round(2)
        
        inning_summary.columns = ['Avg Velocity', 'At Bats', 'Total Pitches']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pitches = px.bar(
                x=inning_summary.index,
                y=inning_summary['Total Pitches'],
                title="Pitches per Inning"
            )
            fig_pitches.update_xaxes(title="Inning")
            fig_pitches.update_yaxes(title="Number of Pitches")
            st.plotly_chart(fig_pitches, use_container_width=True)
        
        with col2:
            fig_velocity_trend = px.line(
                x=inning_summary.index,
                y=inning_summary['Avg Velocity'],
                title="Average Velocity by Inning"
            )
            fig_velocity_trend.update_xaxes(title="Inning")
            fig_velocity_trend.update_yaxes(title="Average Velocity (mph)")
            st.plotly_chart(fig_velocity_trend, use_container_width=True)
            
    def render_defensive_coaching_analysis(self, data):
        """Render defensive coaching insights."""
        st.header("🛡️ Defensive Coaching Analytics")
        
        # Support uploading tracking/expected-position files (CSV or Parquet)
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
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                tmp.write(tracking_file.getvalue())
                tmp.flush()
                tracking_df = pd.read_parquet(tmp.name)
                tmp.close()
            else:
                tracking_df = pd.read_csv(tracking_file)
        
        if expected_file is not None:
            if expected_file.name.lower().endswith(".parquet"):
                import tempfile
                tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                tmp2.write(expected_file.getvalue())
                tmp2.flush()
                expected_df = pd.read_parquet(tmp2.name)
                tmp2.close()
            else:
                expected_df = pd.read_csv(expected_file)
        
        from defensive_analytics import DefensiveAnalytics
        # DefensiveAnalytics may not accept tracking kwargs in __init__; instantiate with data only and
        # try to pass tracking/expected into analysis methods where supported.
        try:
            defensive_analyzer = DefensiveAnalytics(data)
        except TypeError:
            # fallback: try no-arg constructor then attempt to load data via a known loader method
            defensive_analyzer = DefensiveAnalytics()
            if hasattr(defensive_analyzer, 'load_data'):
                try:
                    defensive_analyzer.load_data(data)
                except Exception:
                    pass

        def _safe_call(method_name, *args, **kwargs):
            method = getattr(defensive_analyzer, method_name)
            try:
                return method(*args, **kwargs)
            except TypeError:
                # try without kwargs
                try:
                    return method(*args)
                except Exception:
                    try:
                        return method()
                    except Exception:
                        return {'error': f'failed to call {method_name}'}

        # call analyze_fielder_positioning trying to pass tracking/expected if available
        positioning = _safe_call('analyze_fielder_positioning', tracking_df, expected_df)
         
        if isinstance(positioning, dict) and 'error' in positioning:
            st.error("Positioning analysis failed: " + str(positioning.get('error')))
        else:
             # Create positioning dataframe for display
            pos_df = pd.DataFrame(positioning).T
            st.dataframe(pos_df, use_container_width=True)
            
            # Visual insights
            col1, col2 = st.columns(2)
            
            with col1:
                # Route efficiency by player
                fig_route = px.bar(
                    x=pos_df.index,
                    y=pos_df['avg_route_efficiency'],
                    title="Average Route Efficiency by Fielder",
                    labels={'x': 'Fielder', 'y': 'Route Efficiency (%)'}
                )
                fig_route.add_hline(y=85, line_dash="dash", line_color="green",
                                annotation_text="Target: 85%")
                st.plotly_chart(fig_route, use_container_width=True)
            
            with col2:
                # Reaction time coaching chart
                fig_reaction = px.scatter(
                    x=pos_df['avg_reaction_time'],
                    y=pos_df['avg_route_efficiency'],
                    text=pos_df.index,
                    title="Reaction Time vs Route Efficiency",
                    labels={'x': 'Avg Reaction Time (s)', 'y': 'Route Efficiency (%)'}
                )
                fig_reaction.add_vline(x=pos_df['avg_reaction_time'].mean(), 
                                    line_dash="dash", annotation_text="Team Avg")
                st.plotly_chart(fig_reaction, use_container_width=True)
        
        # Shift validation
        st.subheader("Defensive Shift Effectiveness")
        shift_analysis = defensive_analyzer.validate_defensive_shifts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Probability Catches", shift_analysis['high_probability_catches'])
        with col2:
            st.metric("Missed Opportunities", shift_analysis['missed_opportunities'])
        with col3:
            st.metric("Optimal Positioning Rate", f"{shift_analysis['optimal_positioning_rate']:.1f}%")
        
        # Coaching insights
        st.subheader("Coaching Recommendations")
        insights = defensive_analyzer.reaction_time_coaching_insights()
        
        if insights['players_needing_improvement']:
            st.warning("**Players needing reaction time improvement:**")
            for player in insights['players_needing_improvement']:
                st.write(f"• {player}")
        
        if insights['top_performers']:
            st.success("**Top defensive performers:**")
            for player in insights['top_performers']:
                st.write(f"• {player}")
        
        # Fielding heatmap
        heatmap = defensive_analyzer.create_fielding_heatmap()
        if heatmap:
            st.subheader("Fielding Performance Heatmap")
            st.plotly_chart(heatmap, use_container_width=True)

    def render_coaching_reports(self, data):
        """Generate automated coaching reports."""
        st.header("📋 Coaching Reports & Insights")
        
        # Weekly summary generator
        st.subheader("Game Summary Generator")
        
        if st.button("Generate Coaching Summary"):
            summary = self.generate_coaching_summary(data)
            st.text_area("Coaching Summary", summary, height=400)
            
            # Download button
            st.download_button(
                label="Download Coaching Report",
                data=summary,
                file_name=f"coaching_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

            # Also offer a PDF weekly report generated from the same filtered data
            try:
                rg = ReportGenerator(data)
                pdf_bytes = rg.export_pdf_bytes()
                st.download_button(
                    label="Download Weekly PDF Report",
                    data=pdf_bytes,
                    file_name=f"weekly_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")
        
        # Top 5 insights
        st.subheader("Top 5 Game Insights")
        insights = self.generate_top_insights(data)
        
        for i, insight in enumerate(insights, 1):
            st.info(f"**{i}.** {insight}")

    def generate_coaching_summary(self, data):
        """Generate focused coaching summary."""
        summary = []
        summary.append("=" * 60)
        summary.append("GT BASEBALL COACHING SUMMARY")
        summary.append("=" * 60)
        summary.append("")
        
        # Pitching insights
        summary.append("🥎 PITCHING INSIGHTS:")
        pv = self._numeric_series(data, ['PitchVelo', 'pitch_velo', 'pitch_velocity', 'throw_velo', 'velo'])
        avg_velo = pv.mean()
        summary.append(f"• Team average velocity: {avg_velo:.1f} mph" if not pd.isna(avg_velo) else "• Team average velocity: N/A")
        
        # Strike rate analysis
        strikes = data['PitchOutcome'].isin(['Strike', 'Foul']).sum()
        strike_rate = strikes / len(data) * 100
        summary.append(f"• Strike rate: {strike_rate:.1f}% {'✅' if strike_rate > 65 else '⚠️'}")
        
        # Individual pitcher focus
        # safe pitcher stats using available name/velocity columns
        pitcher_col = self._find_col(data, ['PitcherName', 'pitcher', 'pitcher_name', 'player_name', 'name'])
        if pitcher_col:
            pitcher_stats = data.groupby(pitcher_col).apply(
                lambda g: pd.Series({
                    'PitchVelo_mean': self._numeric_series(g, ['PitchVelo', 'pitch_velo', 'pitch_velocity', 'throw_velo', 'velo']).mean(),
                    'StrikeRate': (g['PitchOutcome'].isin(['Strike', 'Foul']).sum() / max(1, len(g))) * 100 if 'PitchOutcome' in g.columns else pd.NA
                })
            ).round(1)
        else:
            pitcher_stats = pd.DataFrame()

        if not pitcher_stats.empty:
            # handle different column names from aggregation
            strike_col = [c for c in pitcher_stats.columns if 'Strike' in c or 'PitchOutcome' in c][0]
            lowest = pitcher_stats.loc[pitcher_stats[strike_col].idxmin()]
            summary.append(f"• Focus area: {lowest.name} - {lowest[strike_col]:.1f}% strike rate")
        summary.append("")
        
        # Hitting insights
        summary.append("🏏 HITTING INSIGHTS:")
        hit_data = data[data['BallInPlay'].fillna(False).astype(bool)]
        
        if len(hit_data) > 0:
            avg_exit_velo = hit_data['ExitVelo'].mean()
            hard_hits = (hit_data['ExitVelo'] > 95).sum()
            summary.append(f"• Average exit velocity: {avg_exit_velo:.1f} mph")
            summary.append(f"• Hard hit balls (>95 mph): {hard_hits}")
            
            # Launch angle optimization
            if 'LaunchAng' in hit_data.columns:
                optimal_la = hit_data[(hit_data['LaunchAng'] >= 8) & (hit_data['LaunchAng'] <= 32)]
                summary.append(f"• Optimal launch angles (8-32°): {len(optimal_la)} of {len(hit_data)}")
        
        summary.append("")
        
        # Defensive insights
        summary.append("🛡️ DEFENSIVE INSIGHTS:")
        field_data = data[(data['IsEventPlayer'] == True) & (data['EventPlayerName'].notna())]
        
        if len(field_data) > 0:
            # tolerate multiple fielder route column names
            fre = self._numeric_series(field_data, ['FielderRouteEfficiency', 'route_efficiency', 'FielderRoute'])
            avg_route_eff = fre.mean()
            summary.append(f"• Team route efficiency: {avg_route_eff:.1f}%" if not pd.isna(avg_route_eff) else "• Team route efficiency: N/A")
            
            slow_reactions = field_data[field_data['FielderReaction'] > field_data['FielderReaction'].quantile(0.75)]
            if len(slow_reactions) > 0:
                summary.append(f"• Players needing reaction training: {len(slow_reactions)}")
        
        return "\n".join(summary)

    def generate_top_insights(self, data):
        """Generate top 5 actionable insights."""
        insights = []
        
        # Pitching insights (use available velocity column)
        pv = self._numeric_series(data, ['PitchVelo', 'pitch_velo', 'pitch_velocity', 'throw_velo', 'velo'])
        try:
            if pv.std() > 5:
                insights.append("High velocity variance detected - work on consistent mechanics")
        except Exception:
            pass
        
        # Strike rate insight
        strikes = data['PitchOutcome'].isin(['Strike', 'Foul']).sum()
        strike_rate = strikes / len(data) * 100
        if strike_rate < 60:
            insights.append(f"Strike rate below target (currently {strike_rate:.1f}%) - focus on command")
        
        # Exit velocity insight
        hit_data = data[data['BallInPlay'].fillna(False).astype(bool)]
        if len(hit_data) > 0:
            hard_hit_rate = (hit_data['ExitVelo'] > 95).sum() / len(hit_data) * 100
            if hard_hit_rate > 40:
                insights.append(f"High hard-hit rate ({hard_hit_rate:.1f}%) - review pitch selection/location")
        
        # Fielding insight
        field_data = data[(data['IsEventPlayer'] == True) & (data['EventPlayerName'].notna())]
        if len(field_data) > 0:
            low_efficiency = field_data[field_data['FielderRouteEfficiency'] < 80]
            if len(low_efficiency) > 0:
                insights.append(f"{len(low_efficiency)} fielding plays below 80% efficiency - positioning review needed")
        
        # Speed insight
        if 'BatterTimeToFirst' in hit_data.columns:
            avg_time = hit_data['BatterTimeToFirst'].mean()
            if avg_time > 4.5:
                insights.append(f"Average time to first base: {avg_time:.2f}s - speed training opportunity")
        
        return insights[:5]  # Return top 5 insights  
     
    def render_report_generation(self, data):
        """Render the report generation button and handle download options."""
        st.header("📋 Weekly Report Generation")
        
        if st.button("Generate Weekly Report"):
            # Generate the report (CSV + PDF)
            report_generator = ReportGenerator(data)
            report_generator.export_csv("weekly_summary.csv")
            report_generator.export_pdf("weekly_report.pdf")
            
            st.success("Reports Generated! Download the reports below.")
            
            # Provide download options
            with open("reports/weekly_summary.csv", "rb") as file:
                st.download_button(
                    label="Download CSV Report",
                    data=file,
                    file_name="weekly_summary.csv",
                    mime="text/csv"
                )
            
            with open("reports/weekly_report.pdf", "rb") as file:
                st.download_button(
                    label="Download PDF Report",
                    data=file,
                    file_name="weekly_report.pdf",
                    mime="application/pdf"
                )
     

    def render_video_analysis_prep(self, data):
        """Prepare data for video analysis overlay."""
        st.header("🎥 Video Analysis Preparation")
        
        st.info("This section prepares highlight data for video overlay integration")
        
        # High-value plays for video review
        st.subheader("Plays Worth Video Review")
        
        # Hard hit balls
        hard_hits = data[(data['ExitVelo'] > 95) & (data['BallInPlay'].fillna(False).astype(bool))]
        if len(hard_hits) > 0:
            st.write("**Hard Hit Balls (>95 mph):**")
            video_data = hard_hits[['Inning', 'AtBat', 'BatterName', 'ExitVelo', 'LaunchAng', 'Result']].round(1)
            st.dataframe(video_data, use_container_width=True)
        
        # Exceptional fielding plays
        field_data = data[
            (pd.to_numeric(data['FielderProbability'], errors='coerce').fillna(100) < 30) |
            (pd.to_numeric(data['FielderRouteEfficiency'], errors='coerce').fillna(0) > 95)
        ]
        if len(field_data) > 0:
            st.write("**Exceptional Fielding Plays:**")
            field_video = field_data[['Inning', 'AtBat', 'EventPlayerName', 'FielderProbability', 'FielderRouteEfficiency']].round(1)
            st.dataframe(field_video, use_container_width=True)
        
        # Export for video software
        if st.button("Export Video Analysis Data"):
            export_data = pd.concat([hard_hits, field_data]).drop_duplicates()
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download Video Analysis CSV",
                data=csv,
                file_name="video_analysis_highlights.csv",
                mime="text/csv"
            )

    def render_accountability_metrics(self, data):
        """Render accountability metrics dashboard."""
        st.header("📊 Accountability Metrics")
        
        from accountability_analytics import AccountabilityAnalytics
        accountability = AccountabilityAnalytics(data)
        
        # Configuration section
        with st.expander("⚙️ Configure Standards"):
            st.write("**Baserunning Standards:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.number_input("Secondary Lead - 1B (ft)", value=16.0, key="sec_1b")
            with col2:
                st.number_input("Secondary Lead - 2B (ft)", value=16.0, key="sec_2b")
            with col3:
                st.number_input("Secondary Lead - 3B (ft)", value=14.0, key="sec_3b")
            
            st.write("**Defensive Standards:**")
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Route Efficiency Threshold (%)", value=85.0, key="route_threshold")
            with col2:
                st.number_input("Reaction Time Threshold (s)", value=0.8, key="reaction_threshold")
        
        # Tabs for different accountability views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Baserunning", "Defensive Positioning", "Violations Report", "Team Summary"
        ])
        
        with tab1:
            st.subheader("🏃 Baserunning Accountability")
            
            baserunning_analysis = accountability.analyze_baserunning_accountability()
            
            if 'error' not in baserunning_analysis:
                # Team summary
                if 'team_summary' in baserunning_analysis:
                    summary = baserunning_analysis['team_summary']
                    
                    st.write("**Team Performance:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Opportunities",
                            summary.get('total_opportunities', 0)
                        )
                    with col2:
                        st.metric(
                            "Avg Secondary Lead",
                            f"{summary.get('avg_secondary_lead', 0):.1f} ft"
                        )
                    with col3:
                        st.metric(
                            "Avg Max Speed",
                            f"{summary.get('avg_max_speed', 0):.1f} mph"
                        )
                    
                    # Compliance by base
                    if 'compliance_metrics' in summary:
                        st.write("**Compliance by Base:**")
                        
                        compliance_data = []
                        for base, metrics in summary['compliance_metrics'].items():
                            compliance_data.append({
                                'Base': base.replace('base_', '') + 'B',
                                'Expected Lead (ft)': metrics['expected_lead'],
                                'Actual Avg (ft)': round(metrics['avg_lead'], 1),
                                'Compliant Plays': metrics['compliant'],
                                'Total Plays': metrics['total'],
                                'Compliance Rate': f"{metrics['compliance_rate']:.1f}%"
                            })
                        
                        compliance_df = pd.DataFrame(compliance_data)
                        st.dataframe(compliance_df, use_container_width=True)
                        
                        # Visualization
                        charts = accountability._create_baserunning_charts()
                        if 'secondary_lead_comparison' in charts:
                            st.plotly_chart(charts['secondary_lead_comparison'], use_container_width=True)
                
                # Individual player analysis
                if 'players' in baserunning_analysis and baserunning_analysis['players']:
                    st.write("**Individual Player Analysis:**")
                    
                    player = st.selectbox(
                        "Select Player",
                        list(baserunning_analysis['players'].keys())
                    )
                    
                    if player:
                        player_data = baserunning_analysis['players'][player]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Total Opportunities",
                                player_data['total_opportunities']
                            )
                            st.metric(
                                "Compliance Rate",
                                f"{player_data.get('compliance_rate', 0):.1f}%"
                            )
                        
                        with col2:
                            if 'max_speed' in player_data and player_data['max_speed']:
                                st.metric(
                                    "Avg Max Speed",
                                    f"{player_data['max_speed'].get('actual_avg', 0):.1f} mph"
                                )
                                st.metric(
                                    "Max Speed Achieved",
                                    f"{player_data['max_speed'].get('max_achieved', 0):.1f} mph"
                                )
                        
                        # Secondary lead details
                        if 'secondary_lead' in player_data and player_data['secondary_lead']:
                            st.write("**Secondary Lead Performance:**")
                            
                            lead_data = []
                            for base, metrics in player_data['secondary_lead'].items():
                                lead_data.append({
                                    'Base': base.replace('base_', '') + 'B',
                                    'Expected (ft)': metrics['expected'],
                                    'Actual Avg (ft)': round(metrics['actual_avg'], 1),
                                    'Variance (ft)': round(metrics['variance'], 1),
                                    'Compliant': metrics['compliance_count'],
                                    'Total': metrics['opportunities']
                                })
                            
                            lead_df = pd.DataFrame(lead_data)
                            st.dataframe(lead_df, use_container_width=True)
        
        with tab2:
            st.subheader("🛡️ Defensive Positioning Accountability")
            
            defensive_analysis = accountability.analyze_defensive_positioning_accountability()
            
            if 'error' not in defensive_analysis:
                # Team summary
                if 'team_summary' in defensive_analysis:
                    summary = defensive_analysis['team_summary']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Plays",
                            summary.get('total_opportunities', 0)
                        )
                    with col2:
                        st.metric(
                            "Avg Route Efficiency",
                            f"{summary.get('avg_route_efficiency', 0):.1f}%"
                        )
                    with col3:
                        st.metric(
                            "Avg Reaction Time",
                            f"{summary.get('avg_reaction_time', 0):.2f}s"
                        )
                    
                    # Compliance metrics
                    if 'compliance_metrics' in summary:
                        st.write("**Team Compliance:**")
                        
                        for metric_name, metrics in summary['compliance_metrics'].items():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**{metric_name.replace('_', ' ').title()}:**")
                                st.progress(metrics['compliance_rate'] / 100)
                            
                            with col2:
                                st.metric(
                                    "Rate",
                                    f"{metrics['compliance_rate']:.1f}%"
                                )
                
                # Individual fielder analysis
                if 'players' in defensive_analysis and defensive_analysis['players']:
                    st.write("**Individual Fielder Analysis:**")
                    
                    fielder = st.selectbox(
                        "Select Fielder",
                        list(defensive_analysis['players'].keys())
                    )
                    
                    if fielder:
                        fielder_data = defensive_analysis['players'][fielder]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Total Plays",
                                fielder_data['total_opportunities']
                            )
                        with col2:
                            st.metric(
                                "Compliance Rate",
                                f"{fielder_data.get('compliance_rate', 0):.1f}%"
                            )
                        with col3:
                            if 'route_efficiency' in fielder_data:
                                violations = fielder_data['route_efficiency'].get('below_threshold_count', 0)
                                st.metric(
                                    "Violations",
                                    violations,
                                    delta=f"-{violations}" if violations > 0 else None,
                                    delta_color="inverse"
                                )
                        
                        # Detailed metrics
                        if 'route_efficiency' in fielder_data:
                            st.write("**Route Efficiency:**")
                            re = fielder_data['route_efficiency']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"Expected: {re['expected']:.1f}%")
                            with col2:
                                st.write(f"Actual Avg: {re['actual_avg']:.1f}%")
                            with col3:
                                best_play = re.get('best_play')
                                if best_play is not None and not pd.isna(best_play):
                                    st.write(f"Best: {best_play:.1f}%")
                                else:
                                    st.write("Best: N/A")

                        if 'reaction_time' in fielder_data:
                            st.write("**Reaction Time:**")
                            rt = fielder_data['reaction_time']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"Expected: {rt['expected']:.2f}s")
                            with col2:
                                st.write(f"Actual Avg: {rt['actual_avg']:.2f}s")
                            with col3:
                                best_reaction = rt.get('best_reaction')
                                if best_reaction is not None and not pd.isna(best_reaction):
                                    st.write(f"Best: {best_reaction:.2f}s")
                                else:
                                    st.write("Best: N/A")
                    
                    # Visualization
                    charts = accountability._create_defensive_charts()
                    if 'route_efficiency_by_player' in charts:
                        st.plotly_chart(charts['route_efficiency_by_player'], use_container_width=True)
        
        with tab3:
            st.subheader("⚠️ Violations Report")
            
            violations_df = accountability.generate_violation_report()
            
            if not violations_df.empty:
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Violations", len(violations_df))
                with col2:
                    high_severity = len(violations_df[violations_df.get('Severity', '') == 'High'])
                    st.metric("High Severity", high_severity)
                with col3:
                    medium_severity = len(violations_df[violations_df.get('Severity', '') == 'Medium'])
                    st.metric("Medium Severity", medium_severity)
                
                # Filters
                st.write("**Filter Violations:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    violation_type = st.multiselect(
                        "Type",
                        options=violations_df['Type'].unique() if 'Type' in violations_df else [],
                        default=violations_df['Type'].unique() if 'Type' in violations_df else []
                    )
                
                with col2:
                    severity = st.multiselect(
                        "Severity",
                        options=violations_df['Severity'].unique() if 'Severity' in violations_df else [],
                        default=violations_df['Severity'].unique() if 'Severity' in violations_df else []
                    )
                
                # Apply filters
                filtered_violations = violations_df.copy()
                if 'Type' in filtered_violations and violation_type:
                    filtered_violations = filtered_violations[
                        filtered_violations['Type'].isin(violation_type)
                    ]
                if 'Severity' in filtered_violations and severity:
                    filtered_violations = filtered_violations[
                        filtered_violations['Severity'].isin(severity)
                    ]
                
                # Display table
                st.dataframe(filtered_violations, use_container_width=True)
                
                # Download button
                csv = filtered_violations.to_csv(index=False)
                st.download_button(
                    label="Download Violations Report",
                    data=csv,
                    file_name=f"accountability_violations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ No violations found! Team is meeting all standards.")
        
        with tab4:
            st.subheader("📈 Team Summary")
            
            # Overall compliance scorecard
            st.write("**Overall Team Accountability Scorecard:**")
            
            baserunning_analysis = accountability.analyze_baserunning_accountability()
            defensive_analysis = accountability.analyze_defensive_positioning_accountability()
            
            # Calculate overall scores
            baserunning_score = 0
            defensive_score = 0
            
            if 'team_summary' in baserunning_analysis:
                if 'compliance_metrics' in baserunning_analysis['team_summary']:
                    rates = [
                        m['compliance_rate']
                        for m in baserunning_analysis['team_summary']['compliance_metrics'].values()
                    ]
                    baserunning_score = sum(rates) / len(rates) if rates else 0
            
            if 'team_summary' in defensive_analysis:
                if 'compliance_metrics' in defensive_analysis['team_summary']:
                    rates = [
                        m['compliance_rate']
                        for m in defensive_analysis['team_summary']['compliance_metrics'].values()
                    ]
                    defensive_score = sum(rates) / len(rates) if rates else 0
            
            overall_score = (baserunning_score + defensive_score) / 2
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Baserunning Compliance",
                    f"{baserunning_score:.1f}%",
                    delta=f"{baserunning_score - 85:.1f}%" if baserunning_score > 0 else None
                )
            
            with col2:
                st.metric(
                    "Defensive Compliance",
                    f"{defensive_score:.1f}%",
                    delta=f"{defensive_score - 85:.1f}%" if defensive_score > 0 else None
                )
            
            with col3:
                st.metric(
                    "Overall Score",
                    f"{overall_score:.1f}%",
                    delta=f"{overall_score - 85:.1f}%" if overall_score > 0 else None
                )
            
            # Progress bars
            st.write("**Detailed Breakdown:**")
            
            st.write("Baserunning:")
            st.progress(baserunning_score / 100)
            
            st.write("Defensive Positioning:")
            st.progress(defensive_score / 100)

    def run_dashboard(self):
        """Main dashboard execution."""
        data = self.load_data()
        
        if data is None:
            return
        
        # Sidebar filters
        filtered_data, selected_pitcher, selected_batter = self.render_sidebar(data)
        
        # Main content
        self.render_overview_metrics(filtered_data)
        
        # Analysis tabs - Updated with Accountability
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_db = st.tabs([
            "Pitching", "Hitting", "Defensive Analytics", "Baserunning", 
            "Accountability", "Game Flow", "Coaching Reports", "Video Analysis", "Database"
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

# Run the dashboard
if __name__ == "__main__":
    dashboard = GTBaseballDashboard()
    dashboard.run_dashboard()