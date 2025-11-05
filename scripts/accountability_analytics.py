import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import streamlit as st

class AccountabilityAnalytics:
    """
    Accountability metrics for baserunning and defensive positioning.
    Tracks whether players meet expected standards.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
        # Define expected standards (configurable)
        self.standards = {
            'baserunning': {
                'initial_lead_1B': 12.0,  # feet
                'initial_lead_2B': 12.0,
                'initial_lead_3B': 10.0,
                'secondary_lead_1B': 16.0,
                'secondary_lead_2B': 16.0,
                'secondary_lead_3B': 14.0,
                'max_speed_threshold': 27.0,  # mph
            },
            'defensive': {
                'route_efficiency_threshold': 85.0,  # percentage
                'reaction_time_threshold': 0.8,  # seconds
            }
        }
    
    def analyze_baserunning_accountability(self) -> Dict:
        """Analyze baserunning accountability metrics."""
        baserunning_data = self.data[
            (self.data['BaserunnerInitial'].notna()) |
            (self.data['BaserunnerSecondary'].notna()) |
            (self.data['BaserunnerFinal'].notna())
        ].copy()
        
        if len(baserunning_data) == 0:
            return {'error': 'No baserunning data available'}
        
        results = {
            'players': {},
            'team_summary': {},
            'violations': []
        }
        
        # Analyze by runner
        if 'BaserunnerName' in baserunning_data.columns:
            for runner in baserunning_data['BaserunnerName'].unique():
                runner_data = baserunning_data[
                    baserunning_data['BaserunnerName'] == runner
                ]
                results['players'][runner] = self._analyze_runner(runner_data)
        else:
            # If no BaserunnerName, analyze by play
            results['plays'] = self._analyze_baserunning_plays(baserunning_data)
        
        # Team summary
        results['team_summary'] = self._calculate_team_baserunning_summary(baserunning_data)
        
        return results
    
    def _analyze_runner(self, runner_data: pd.DataFrame) -> Dict:
        """Analyze individual runner accountability."""
        analysis = {
            'total_opportunities': len(runner_data),
            'initial_lead': {},
            'secondary_lead': {},
            'max_speed': {},
            'compliance_rate': 0.0
        }
        
        # Initial lead analysis by base
        for base in [1, 2, 3]:
            base_data = runner_data[runner_data['BaserunnerInitial'] == base]
            if len(base_data) > 0:
                expected = self.standards['baserunning'].get(f'initial_lead_{base}B', 12.0)
                # Note: Assuming lead distance data would be in a column
                # For now using BaserunnerInitial as a proxy
                analysis['initial_lead'][f'base_{base}'] = {
                    'opportunities': len(base_data),
                    'expected': expected,
                    'actual_avg': base_data['BaserunnerInitial'].mean(),
                }
        
        # Secondary lead analysis
        secondary_data = runner_data[runner_data['BaserunnerSecondary'].notna()]
        if len(secondary_data) > 0:
            for base in [1, 2, 3]:
                base_secondary = secondary_data[
                    secondary_data['BaserunnerInitial'] == base
                ]
                if len(base_secondary) > 0:
                    expected_secondary = self.standards['baserunning'].get(
                        f'secondary_lead_{base}B', 16.0
                    )
                    actual_secondary = base_secondary['BaserunnerSecondary'].mean()
                    
                    analysis['secondary_lead'][f'base_{base}'] = {
                        'opportunities': len(base_secondary),
                        'expected': expected_secondary,
                        'actual_avg': actual_secondary,
                        'variance': actual_secondary - expected_secondary,
                        'compliance_count': len(
                            base_secondary[
                                base_secondary['BaserunnerSecondary'] >= expected_secondary
                            ]
                        )
                    }
        
        # Max speed analysis
        speed_data = runner_data[runner_data['BaserunnerMaxSpeed'].notna()]
        if len(speed_data) > 0:
            expected_speed = self.standards['baserunning']['max_speed_threshold']
            analysis['max_speed'] = {
                'opportunities': len(speed_data),
                'expected': expected_speed,
                'actual_avg': speed_data['BaserunnerMaxSpeed'].mean(),
                'max_achieved': speed_data['BaserunnerMaxSpeed'].max(),
                'below_threshold_count': len(
                    speed_data[speed_data['BaserunnerMaxSpeed'] < expected_speed]
                )
            }
        
        # Calculate overall compliance rate
        total_checks = 0
        passed_checks = 0
        
        for base, lead_info in analysis['secondary_lead'].items():
            total_checks += lead_info['opportunities']
            passed_checks += lead_info['compliance_count']
        
        if total_checks > 0:
            analysis['compliance_rate'] = (passed_checks / total_checks) * 100
        
        return analysis
    
    def _analyze_baserunning_plays(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze individual baserunning plays."""
        plays = []
        
        for idx, row in data.iterrows():
            play = {
                'inning': row.get('Inning'),
                'at_bat': row.get('AtBat'),
                'initial_base': row.get('BaserunnerInitial'),
                'secondary_lead': row.get('BaserunnerSecondary'),
                'final_base': row.get('BaserunnerFinal'),
                'max_speed': row.get('BaserunnerMaxSpeed'),
                'violations': []
            }
            
            # Check secondary lead compliance
            if pd.notna(play['secondary_lead']) and pd.notna(play['initial_base']):
                expected = self.standards['baserunning'].get(
                    f'secondary_lead_{int(play["initial_base"])}B', 16.0
                )
                if play['secondary_lead'] < expected:
                    play['violations'].append({
                        'metric': 'Secondary Lead',
                        'expected': expected,
                        'actual': play['secondary_lead'],
                        'difference': play['secondary_lead'] - expected
                    })
            
            # Check max speed compliance
            if pd.notna(play['max_speed']):
                expected_speed = self.standards['baserunning']['max_speed_threshold']
                if play['max_speed'] < expected_speed:
                    play['violations'].append({
                        'metric': 'Max Speed',
                        'expected': expected_speed,
                        'actual': play['max_speed'],
                        'difference': play['max_speed'] - expected_speed
                    })
            
            if play['violations']:
                plays.append(play)
        
        return plays
    
    def _calculate_team_baserunning_summary(self, data: pd.DataFrame) -> Dict:
        """Calculate team-wide baserunning summary."""
        summary = {
            'total_opportunities': len(data),
            'avg_secondary_lead': data['BaserunnerSecondary'].mean(),
            'avg_max_speed': data['BaserunnerMaxSpeed'].mean(),
            'compliance_metrics': {}
        }
        
        # Secondary lead compliance by base
        for base in [1, 2, 3]:
            base_data = data[
                (data['BaserunnerInitial'] == base) &
                (data['BaserunnerSecondary'].notna())
            ]
            if len(base_data) > 0:
                expected = self.standards['baserunning'].get(f'secondary_lead_{base}B', 16.0)
                compliant = len(base_data[base_data['BaserunnerSecondary'] >= expected])
                
                summary['compliance_metrics'][f'base_{base}'] = {
                    'total': len(base_data),
                    'compliant': compliant,
                    'compliance_rate': (compliant / len(base_data)) * 100,
                    'avg_lead': base_data['BaserunnerSecondary'].mean(),
                    'expected_lead': expected
                }
        
        return summary
    
    def analyze_defensive_positioning_accountability(self) -> Dict:
        """Analyze defensive positioning accountability."""
        defensive_data = self.data[
            (self.data['IsEventPlayer'] == True) &
            (self.data['EventPlayerName'].notna())
        ].copy()
        
        if len(defensive_data) == 0:
            return {'error': 'No defensive data available'}
        
        results = {
            'players': {},
            'team_summary': {},
            'violations': []
        }
        
        # Analyze by fielder
        for fielder in defensive_data['EventPlayerName'].unique():
            fielder_data = defensive_data[
                defensive_data['EventPlayerName'] == fielder
            ]
            results['players'][fielder] = self._analyze_fielder_accountability(fielder_data)
        
        # Team summary
        results['team_summary'] = self._calculate_team_defensive_summary(defensive_data)
        
        return results
    
    def _analyze_fielder_accountability(self, fielder_data: pd.DataFrame) -> Dict:
        """Analyze individual fielder accountability."""
        analysis = {
            'total_opportunities': len(fielder_data),
            'route_efficiency': {},
            'reaction_time': {},
            'compliance_rate': 0.0
        }
        
        # Route efficiency analysis
        if 'FielderRouteEfficiency' in fielder_data.columns:
            route_data = fielder_data['FielderRouteEfficiency'].dropna()
            expected_efficiency = self.standards['defensive']['route_efficiency_threshold']
            
            analysis['route_efficiency'] = {
                'opportunities': len(route_data),
                'expected': expected_efficiency,
                'actual_avg': route_data.mean() if len(route_data) > 0 else 0,
                'below_threshold_count': len(route_data[route_data < expected_efficiency]),
                'worst_play': float(route_data.min()) if len(route_data) > 0 else None,
                'best_play': float(route_data.max()) if len(route_data) > 0 else None
            }
        
        # Reaction time analysis
        if 'FielderReaction' in fielder_data.columns:
            reaction_data = fielder_data['FielderReaction'].dropna()
            expected_reaction = self.standards['defensive']['reaction_time_threshold']
            
            analysis['reaction_time'] = {
                'opportunities': len(reaction_data),
                'expected': expected_reaction,
                'actual_avg': reaction_data.mean() if len(reaction_data) > 0 else 0,
                'above_threshold_count': len(reaction_data[reaction_data > expected_reaction]),
                'best_reaction': float(reaction_data.min()) if len(reaction_data) > 0 else None,
                'worst_reaction': float(reaction_data.max()) if len(reaction_data) > 0 else None
            }
        
        # Calculate compliance rate
        total_checks = 0
        passed_checks = 0
        
        if 'route_efficiency' in analysis and analysis['route_efficiency'].get('opportunities', 0) > 0:
            total_checks += analysis['route_efficiency']['opportunities']
            passed_checks += (
                analysis['route_efficiency']['opportunities'] -
                analysis['route_efficiency']['below_threshold_count']
            )
        
        if 'reaction_time' in analysis and analysis['reaction_time'].get('opportunities', 0) > 0:
            total_checks += analysis['reaction_time']['opportunities']
            passed_checks += (
                analysis['reaction_time']['opportunities'] -
                analysis['reaction_time']['above_threshold_count']
            )
        
        if total_checks > 0:
            analysis['compliance_rate'] = (passed_checks / total_checks) * 100
        
        return analysis
    
    def _calculate_team_defensive_summary(self, data: pd.DataFrame) -> Dict:
        """Calculate team-wide defensive summary."""
        summary = {
            'total_opportunities': len(data),
            'avg_route_efficiency': data['FielderRouteEfficiency'].mean(),
            'avg_reaction_time': data['FielderReaction'].mean(),
            'compliance_metrics': {}
        }
        
        # Route efficiency compliance
        if 'FielderRouteEfficiency' in data.columns:
            route_data = data['FielderRouteEfficiency'].dropna()
            expected = self.standards['defensive']['route_efficiency_threshold']
            compliant = len(route_data[route_data >= expected])
            
            summary['compliance_metrics']['route_efficiency'] = {
                'total': len(route_data),
                'compliant': compliant,
                'compliance_rate': (compliant / len(route_data)) * 100 if len(route_data) > 0 else 0,
                'expected': expected
            }
        
        # Reaction time compliance
        if 'FielderReaction' in data.columns:
            reaction_data = data['FielderReaction'].dropna()
            expected = self.standards['defensive']['reaction_time_threshold']
            compliant = len(reaction_data[reaction_data <= expected])
            
            summary['compliance_metrics']['reaction_time'] = {
                'total': len(reaction_data),
                'compliant': compliant,
                'compliance_rate': (compliant / len(reaction_data)) * 100 if len(reaction_data) > 0 else 0,
                'expected': expected
            }
        
        return summary
    
    def create_accountability_dashboard_charts(self, metric_type: str = 'baserunning'):
        """Create visualizations for accountability metrics."""
        if metric_type == 'baserunning':
            return self._create_baserunning_charts()
        else:
            return self._create_defensive_charts()
    
    def _create_baserunning_charts(self) -> Dict:
        """Create baserunning accountability charts."""
        charts = {}
        
        baserunning_data = self.data[
            self.data['BaserunnerSecondary'].notna()
        ].copy()
        
        if len(baserunning_data) == 0:
            return charts
        
        # Secondary lead compliance chart
        compliance_data = []
        for base in [1, 2, 3]:
            base_data = baserunning_data[baserunning_data['BaserunnerInitial'] == base]
            if len(base_data) > 0:
                expected = self.standards['baserunning'].get(f'secondary_lead_{base}B', 16.0)
                actual = base_data['BaserunnerSecondary'].mean()
                
                compliance_data.append({
                    'Base': f'{base}B',
                    'Expected': expected,
                    'Actual': actual,
                    'Difference': actual - expected
                })
        
        if compliance_data:
            df = pd.DataFrame(compliance_data)
            
            # Create comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Expected',
                x=df['Base'],
                y=df['Expected'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Actual',
                x=df['Base'],
                y=df['Actual'],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Secondary Lead: Expected vs Actual (feet)',
                barmode='group',
                yaxis_title='Lead Distance (feet)',
                xaxis_title='Base'
            )
            
            charts['secondary_lead_comparison'] = fig
        
        return charts
    
    def _create_defensive_charts(self) -> Dict:
        """Create defensive accountability charts."""
        charts = {}
        
        defensive_data = self.data[
            (self.data['IsEventPlayer'] == True) &
            (self.data['EventPlayerName'].notna())
        ].copy()
        
        if len(defensive_data) == 0:
            return charts
        
        # Route efficiency by player
        if 'FielderRouteEfficiency' in defensive_data.columns:
            player_efficiency = defensive_data.groupby('EventPlayerName')[
                'FielderRouteEfficiency'
            ].mean().sort_values()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=player_efficiency.values,
                y=player_efficiency.index,
                orientation='h',
                marker_color=['red' if x < self.standards['defensive']['route_efficiency_threshold'] 
                             else 'green' for x in player_efficiency.values]
            ))
            
            fig.add_vline(
                x=self.standards['defensive']['route_efficiency_threshold'],
                line_dash="dash",
                line_color="black",
                annotation_text="Target"
            )
            
            fig.update_layout(
                title='Route Efficiency by Player',
                xaxis_title='Route Efficiency (%)',
                yaxis_title='Player'
            )
            
            charts['route_efficiency_by_player'] = fig
        
        return charts
    
    def generate_violation_report(self) -> pd.DataFrame:
        """Generate a report of all accountability violations."""
        violations = []
        
        # Baserunning violations
        baserunning_analysis = self.analyze_baserunning_accountability()
        if 'plays' in baserunning_analysis:
            for play in baserunning_analysis['plays']:
                for violation in play['violations']:
                    violations.append({
                        'Type': 'Baserunning',
                        'Inning': play['inning'],
                        'AtBat': play['at_bat'],
                        'Metric': violation['metric'],
                        'Expected': violation['expected'],
                        'Actual': violation['actual'],
                        'Difference': violation['difference'],
                        'Severity': 'High' if abs(violation['difference']) > 2 else 'Medium'
                    })
        
        # Defensive violations
        defensive_analysis = self.analyze_defensive_positioning_accountability()
        if 'players' in defensive_analysis:
            for player, data in defensive_analysis['players'].items():
                if 'route_efficiency' in data:
                    if data['route_efficiency'].get('below_threshold_count', 0) > 0:
                        violations.append({
                            'Type': 'Defensive',
                            'Player': player,
                            'Metric': 'Route Efficiency',
                            'Expected': data['route_efficiency']['expected'],
                            'Actual': data['route_efficiency']['actual_avg'],
                            'Violations': data['route_efficiency']['below_threshold_count'],
                            'Severity': 'High' if data['route_efficiency']['actual_avg'] < 75 else 'Medium'
                        })
        
        return pd.DataFrame(violations) if violations else pd.DataFrame()