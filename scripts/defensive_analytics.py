import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple

class DefensiveAnalytics:
    """
    Advanced defensive analytics for GT Baseball coaching insights.
    """
    
    def __init__(self, data=None):
        """
        Defensive constructor: accept None / list / DataFrame and normalize common
        fielding-related column names so downstream methods don't KeyError.
        """
        import pandas as pd

        # Ensure we always have a DataFrame to work with
        if data is None:
            df = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            try:
                df = pd.DataFrame(data)
            except Exception:
                df = pd.DataFrame()

        # Helper to find a candidate column (case-sensitive check against existing columns)
        def _find(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        # Common variants we expect from various exports
        event_name_col = _find([
            'EventPlayerName', 'eventplayername', 'event_player_name', 'EventPlayer',
            'player_involved', 'player_name', 'name', 'primary_fielder'
        ])
        is_event_col = _find([
            'IsEventPlayer', 'is_event_player', 'IsEvent', 'is_event', 'event_player_flag',
            'is_eventplayer', 'primary_fielder', 'primary_fielder'.lower()
        ])
        route_col = _find(['FielderRouteEfficiency', 'route_efficiency', 'FielderRoute', 'routeEff', 'route_eff'])
        reaction_col = _find(['FielderReaction', 'reaction_time', 'fielder_reaction', 'reaction', 'time_top_speed_fielder'])
        maxspeed_col = _find(['FielderMaxSpeed', 'fielder_max_speed', 'FielderMax', 'max_speed_fielder', 'maxspeed', 'max_speed', 'max_speed_runner', 'max_speed_fielder'])
        prob_col = _find(['FielderProbability', 'FielderProb', 'Probability', 'fielder_probability', 'catch_probability', 'Probability'])
        # additional raw fields
        lead_baserunner_col = _find(['lead_baserunner'])
        time_top_speed_col = _find(['time_top_speed_fielder', 'time_top_speed_runner'])
 
        # Create canonical columns so other code can rely on these names
        if event_name_col:
            df['EventPlayerName'] = df[event_name_col]
        else:
            df['EventPlayerName'] = pd.NA
 
        if is_event_col:
            try:
                df['IsEventPlayer'] = df[is_event_col].astype(bool)
            except Exception:
                df['IsEventPlayer'] = df[is_event_col].notna()
        else:
            # infer IsEventPlayer from presence of an event player name
            df['IsEventPlayer'] = df['EventPlayerName'].notna() & (df['EventPlayerName'].astype(str) != '')
 
        # Standardize route/reaction column existences for downstream usage
        if route_col and 'FielderRouteEfficiency' not in df.columns:
            df['FielderRouteEfficiency'] = df[route_col]
        if reaction_col and 'FielderReaction' not in df.columns:
            df['FielderReaction'] = df[reaction_col]
        # Map max speed and probability if present
        if maxspeed_col and 'FielderMaxSpeed' not in df.columns:
            df['FielderMaxSpeed'] = pd.to_numeric(df[maxspeed_col], errors='coerce')
        else:
            # if parquet contains runner max speed column name, map that too
            if 'max_speed_runner' in df.columns and 'FielderMaxSpeed' not in df.columns:
                df['FielderMaxSpeed'] = pd.to_numeric(df['max_speed_runner'], errors='coerce')
 
        if prob_col and 'FielderProbability' not in df.columns:
            df['FielderProbability'] = pd.to_numeric(df[prob_col], errors='coerce')
 
        # optional mapping for lead baserunner/time_top_speed if present
        if lead_baserunner_col and 'lead_baserunner' not in df.columns:
            df['lead_baserunner'] = df[lead_baserunner_col]
        if time_top_speed_col and 'time_top_speed_fielder' not in df.columns:
            df['time_top_speed_fielder'] = df[time_top_speed_col]
 
        # Keep the normalized DataFrame and a pre-filtered fielding DataFrame
        self.data = df
        try:
            self.fielding_data = df[(df['IsEventPlayer'] == True) & df['EventPlayerName'].notna()]
        except Exception:
            self.fielding_data = pd.DataFrame()
 
        # Other initialization can follow (avoid assuming any other specific column names)
        # ...existing initialization code (if any)...
    
    def analyze_fielder_positioning(self) -> Dict:
        """Analyze fielder positioning effectiveness."""
        if len(self.fielding_data) == 0:
            return {'error': 'No fielding data available'}
        
        positioning_analysis = {}
        
        for fielder in self.fielding_data['EventPlayerName'].unique():
            fielder_data = self.fielding_data[
                self.fielding_data['EventPlayerName'] == fielder
            ]
            
            positioning_analysis[fielder] = {
                'total_plays': len(fielder_data),
                'avg_route_efficiency': fielder_data['FielderRouteEfficiency'].mean(),
                'avg_reaction_time': fielder_data['FielderReaction'].mean(),
                'max_speed_achieved': fielder_data['FielderMaxSpeed'].max(),
                'catch_probability_avg': fielder_data['FielderProbability'].mean(),
                'successful_plays': len(fielder_data[
                    fielder_data['FielderProbability'] > 50
                ])
            }
        
        return positioning_analysis
    
    def validate_defensive_shifts(self) -> Dict:
        """Analyze effectiveness of defensive positioning."""
        shift_analysis = {
            'high_probability_catches': 0,
            'missed_opportunities': 0,
            'optimal_positioning_rate': 0
        }
        
        if 'FielderProbability' in self.fielding_data.columns:
            high_prob = self.fielding_data[self.fielding_data['FielderProbability'] > 70]
            low_efficiency = self.fielding_data[
                (self.fielding_data['FielderProbability'] > 50) &
                (self.fielding_data.get('FielderRouteEfficiency', pd.Series(dtype=float)) < 80)
            ]

            total = len(self.fielding_data)
            # safe ratio: avoid division by zero
            optimal_rate = (len(high_prob) / total * 100) if total > 0 else 0.0

            shift_analysis.update({
                'high_probability_catches': int(len(high_prob)),
                'missed_opportunities': int(len(low_efficiency)),
                'optimal_positioning_rate': float(optimal_rate)
            })
        
        return shift_analysis
    
    def create_fielding_heatmap(self):
        """
        Create a simple fielding performance heatmap per fielder.
        Returns a Plotly figure or None if insufficient data.
        """
        # Build aggregation dict only for columns that exist to avoid KeyError
        agg_map = {}
        if 'FielderRouteEfficiency' in self.fielding_data.columns:
            agg_map['FielderRouteEfficiency'] = 'mean'
        if 'FielderReaction' in self.fielding_data.columns:
            agg_map['FielderReaction'] = 'mean'
        if 'FielderMaxSpeed' in self.fielding_data.columns:
            agg_map['FielderMaxSpeed'] = 'max'
        if 'FielderProbability' in self.fielding_data.columns:
            agg_map['FielderProbability'] = 'mean'

        if not agg_map:
            # Not enough fields to compute a heatmap
            return None

        fielder_metrics = self.fielding_data.groupby('EventPlayerName').agg(agg_map).round(2)
        if fielder_metrics.empty:
            return None

        # Fill NaN -> 0 for visualization but keep original for values display
        viz_values = fielder_metrics.fillna(0).values

        fig = go.Figure(
            data=go.Heatmap(
                z=viz_values,
                x=fielder_metrics.columns.tolist(),
                y=fielder_metrics.index.tolist(),
                colorscale='Viridis',
                hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"
            )
        )
        fig.update_layout(title="Fielding Metrics Heatmap", xaxis_title="", yaxis_title="")
        return fig
    
    def reaction_time_coaching_insights(self) -> Dict:
        """Generate coaching insights for reaction times."""
        insights = {
            'players_needing_improvement': [],
            'top_performers': [],
            'average_benchmarks': {}
        }
        
        if 'FielderReaction' in self.fielding_data.columns:
            reaction_stats = self.fielding_data.groupby('EventPlayerName')['FielderReaction'].agg(['mean', 'count'])
            
            # Players with slow reaction times (bottom 25%)
            slow_threshold = reaction_stats['mean'].quantile(0.75)
            fast_threshold = reaction_stats['mean'].quantile(0.25)
            
            insights['players_needing_improvement'] = reaction_stats[
                reaction_stats['mean'] > slow_threshold
            ].index.tolist()
            
            insights['top_performers'] = reaction_stats[
                reaction_stats['mean'] < fast_threshold
            ].index.tolist()
            
            insights['average_benchmarks'] = {
                'team_average': reaction_stats['mean'].mean(),
                'target_reaction_time': fast_threshold
            }
        
        return insights