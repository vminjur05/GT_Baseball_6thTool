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
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fielding_data = data[(data['IsEventPlayer'] == True) & 
                                 (data['EventPlayerName'].notna())]
    
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
            high_prob = self.fielding_data[
                self.fielding_data['FielderProbability'] > 70
            ]
            low_efficiency = self.fielding_data[
                (self.fielding_data['FielderProbability'] > 50) & 
                (self.fielding_data['FielderRouteEfficiency'] < 80)
            ]
            
            shift_analysis.update({
                'high_probability_catches': len(high_prob),
                'missed_opportunities': len(low_efficiency),
                'optimal_positioning_rate': len(high_prob) / len(self.fielding_data) * 100
            })
        
        return shift_analysis
    
    def create_fielding_heatmap(self):
        """Create fielding efficiency heatmap."""
        if len(self.fielding_data) == 0:
            return None
        
        # Group by fielder and calculate metrics
        fielder_metrics = self.fielding_data.groupby('EventPlayerName').agg({
            'FielderRouteEfficiency': 'mean',
            'FielderReaction': 'mean',
            'FielderMaxSpeed': 'max',
            'FielderProbability': 'mean'
        }).round(2)
        
        # Create heatmap
        fig = px.imshow(
            fielder_metrics.T,
            labels=dict(x="Fielder", y="Metric", color="Value"),
            x=fielder_metrics.index,
            y=['Route Efficiency', 'Reaction Time', 'Max Speed', 'Catch Probability'],
            title="Fielding Performance Heatmap",
            color_continuous_scale="RdYlGn"
        )
        
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