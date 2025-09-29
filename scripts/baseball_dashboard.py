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

class GTBaseballDashboard:
    """
    Interactive Streamlit dashboard for GT Baseball 6th Tool analytics.
    """
    
    def __init__(self):
        self.setup_page_config()
        
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
            loader = GTBaseballDataLoader()
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload GT Baseball CSV files", 
                accept_multiple_files=True,
                type=['csv']
            )
            
            if uploaded_files:
                all_data = []
                for i, file in enumerate(uploaded_files):
                    try:
                        df = pd.read_csv(file)
                        df['GameID'] = f"Game_{i+1}"
                        df['FileName'] = file.name
                        all_data.append(df)
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")
                
                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    # Apply the same cleaning as the loader
                    loader.data = combined_data
                    st.session_state.game_data = loader._clean_data(combined_data)
                    st.session_state.game_data = loader._add_derived_columns(st.session_state.game_data)
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
        
        pitchers = ['All Pitchers'] + list(data['PitcherName'].unique())
        selected_pitcher = st.sidebar.selectbox("Select Pitcher:", pitchers)
        
        batters = ['All Batters'] + list(data['BatterName'].unique())
        selected_batter = st.sidebar.selectbox("Select Batter:", batters)
        
        # Apply filters
        filtered_data = data.copy()
        if selected_pitcher != 'All Pitchers':
            filtered_data = filtered_data[filtered_data['PitcherName'] == selected_pitcher]
        if selected_batter != 'All Batters':
            filtered_data = filtered_data[filtered_data['BatterName'] == selected_batter]
        
        # Inning filter
        innings = sorted(data['Inning'].unique())
        selected_innings = st.sidebar.multiselect(
            "Select Innings:", 
            innings, 
            default=innings
        )
        filtered_data = filtered_data[filtered_data['Inning'].isin(selected_innings)]
        
        return filtered_data, selected_pitcher, selected_batter
    
    def render_overview_metrics(self, data):
        """Render key performance metrics."""
        st.markdown('<div class="main-header">GT Baseball 6th Tool Analytics</div>', unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_pitches = len(data)
            st.metric("Total Pitches", f"{total_pitches:,}")
        
        with col2:
            avg_velocity = data['PitchVelo'].mean()
            st.metric("Avg Pitch Velocity", f"{avg_velocity:.1f} mph")
        
        with col3:
            balls_in_play = data['BallInPlay'].sum()
            st.metric("Balls in Play", f"{balls_in_play:,}")
        
        with col4:
            if 'ExitVelo' in data.columns:
                avg_exit_velo = data[data['BallInPlay'] == True]['ExitVelo'].mean()
                st.metric("Avg Exit Velocity", f"{avg_exit_velo:.1f} mph")
            else:
                st.metric("Avg Exit Velocity", "N/A")
        
        with col5:
            unique_players = len(set(data['PitcherName'].unique()) | set(data['BatterName'].unique()))
            st.metric("Total Players", unique_players)
    
    def render_pitching_analysis(self, data):
        """Render pitching analysis section."""
        st.header("🥎 Pitching Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Velocity distribution
            fig_vel = px.histogram(
                data, 
                x='PitchVelo', 
                nbins=20,
                title="Pitch Velocity Distribution",
                labels={'PitchVelo': 'Velocity (mph)', 'count': 'Frequency'}
            )
            fig_vel.add_vline(x=data['PitchVelo'].mean(), line_dash="dash", 
                             annotation_text=f"Mean: {data['PitchVelo'].mean():.1f}")
            st.plotly_chart(fig_vel, use_container_width=True)
        
        with col2:
            # Pitch outcomes
            if 'PitchOutcome' in data.columns:
                outcome_counts = data['PitchOutcome'].value_counts()
                fig_outcomes = px.pie(
                    values=outcome_counts.values,
                    names=outcome_counts.index,
                    title="Pitch Outcome Distribution"
                )
                st.plotly_chart(fig_outcomes, use_container_width=True)
        
        # Velocity by inning
        if data['Inning'].nunique() > 1:
            st.subheader("Velocity by Inning")
            fig_inning = px.box(
                data,
                x='Inning',
                y='PitchVelo',
                title="Pitch Velocity by Inning"
            )
            st.plotly_chart(fig_inning, use_container_width=True)
        
        # Pitcher comparison
        st.subheader("Pitcher Comparison")
        pitcher_stats = data.groupby('PitcherName').agg({
            'PitchVelo': ['mean', 'count'],
            'PitchOutcome': lambda x: (x == 'Strike').sum() / len(x) * 100
        }).round(2)
        
        pitcher_stats.columns = ['Avg Velocity', 'Total Pitches', 'Strike Rate %']
        st.dataframe(pitcher_stats, use_container_width=True)
    
    def render_hitting_analysis(self, data):
        """Render hitting analysis section."""
        st.header("🏏 Hitting Analysis")
        
        hit_data = data[data['BallInPlay'] == True]
        
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
        
        base_data = data[data['BaserunnerMaxSpeed'].notna()]
        
        if len(base_data) == 0:
            st.warning("No baserunning data available for analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Max speed distribution
            fig_speed = px.histogram(
                base_data,
                x='BaserunnerMaxSpeed',
                title="Baserunner Max Speed Distribution",
                labels={'BaserunnerMaxSpeed': 'Max Speed (mph)'}
            )
            st.plotly_chart(fig_speed, use_container_width=True)
        
        with col2:
            # Speed by base position
            if all(col in base_data.columns for col in ['BaserunnerInitial', 'BaserunnerFinal']):
                base_data_clean = base_data.dropna(subset=['BaserunnerInitial', 'BaserunnerFinal'])
                if len(base_data_clean) > 0:
                    fig_bases = px.box(
                        base_data_clean,
                        x='BaserunnerInitial',
                        y='BaserunnerMaxSpeed',
                        title="Speed by Starting Base"
                    )
                    st.plotly_chart(fig_bases, use_container_width=True)
    
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
    
    # Add this method to GTBaseballDashboard class

    def render_defensive_coaching_analysis(self, data):
        """Render defensive coaching insights."""
        st.header("🛡️ Defensive Coaching Analytics")
        
        from defensive_analytics import DefensiveAnalytics
        defensive_analyzer = DefensiveAnalytics(data)
        
        # Fielder positioning analysis
        st.subheader("Fielder Positioning Effectiveness")
        positioning = defensive_analyzer.analyze_fielder_positioning()
        
        if 'error' not in positioning:
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
        avg_velo = data['PitchVelo'].mean()
        summary.append(f"• Team average velocity: {avg_velo:.1f} mph")
        
        # Strike rate analysis
        strikes = data['PitchOutcome'].isin(['Strike', 'Foul']).sum()
        strike_rate = strikes / len(data) * 100
        summary.append(f"• Strike rate: {strike_rate:.1f}% {'✅' if strike_rate > 65 else '⚠️'}")
        
        # Individual pitcher focus
        pitcher_stats = data.groupby('PitcherName').agg({
            'PitchVelo': 'mean',
            'PitchOutcome': lambda x: (x.isin(['Strike', 'Foul']).sum() / len(x) * 100)
        }).round(1)
        
        lowest_strike_rate = pitcher_stats.loc[pitcher_stats['PitchOutcome'].idxmin()]
        summary.append(f"• Focus area: {lowest_strike_rate.name} - {lowest_strike_rate['PitchOutcome']:.1f}% strike rate")
        summary.append("")
        
        # Hitting insights
        summary.append("🏏 HITTING INSIGHTS:")
        hit_data = data[data['BallInPlay'] == True]
        
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
            avg_route_eff = field_data['FielderRouteEfficiency'].mean()
            summary.append(f"• Team route efficiency: {avg_route_eff:.1f}%")
            
            slow_reactions = field_data[field_data['FielderReaction'] > field_data['FielderReaction'].quantile(0.75)]
            if len(slow_reactions) > 0:
                summary.append(f"• Players needing reaction training: {len(slow_reactions)}")
        
        return "\n".join(summary)

    def generate_top_insights(self, data):
        """Generate top 5 actionable insights."""
        insights = []
        
        # Pitching insights
        if data['PitchVelo'].std() > 5:
            insights.append("High velocity variance detected - work on consistent mechanics")
        
        # Strike rate insight
        strikes = data['PitchOutcome'].isin(['Strike', 'Foul']).sum()
        strike_rate = strikes / len(data) * 100
        if strike_rate < 60:
            insights.append(f"Strike rate below target (currently {strike_rate:.1f}%) - focus on command")
        
        # Exit velocity insight
        hit_data = data[data['BallInPlay'] == True]
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
        hard_hits = data[(data['ExitVelo'] > 95) & (data['BallInPlay'] == True)]
        if len(hard_hits) > 0:
            st.write("**Hard Hit Balls (>95 mph):**")
            video_data = hard_hits[['Inning', 'AtBat', 'BatterName', 'ExitVelo', 'LaunchAng', 'Result']].round(1)
            st.dataframe(video_data, use_container_width=True)
        
        # Exceptional fielding plays
        field_data = data[(data['FielderProbability'] < 30) | (data['FielderRouteEfficiency'] > 95)]
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

    def run_dashboard(self):
        """Main dashboard execution."""
        data = self.load_data()
        
        if data is None:
            return
        
        # Sidebar filters
        filtered_data, selected_pitcher, selected_batter = self.render_sidebar(data)
        
        # Main content
        self.render_overview_metrics(filtered_data)
        
        # Analysis tabs - Updated with coaching focus
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Pitching", "Hitting", "Defensive Analytics", "Baserunning", 
            "Game Flow", "Coaching Reports", "Video Analysis"
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
            self.render_game_flow(filtered_data)
        
        with tab6:
            self.render_coaching_reports(filtered_data)
        
        with tab7:
            self.render_video_analysis_prep(filtered_data)

# Run the dashboard
if __name__ == "__main__":
    dashboard = GTBaseballDashboard()
    dashboard.run_dashboard()