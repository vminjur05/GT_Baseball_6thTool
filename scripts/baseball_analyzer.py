import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GTBaseballAnalyzer:
    """
    Analytics engine for GT Baseball 6th Tool data.
    Provides comprehensive analysis of pitching, hitting, fielding, and baserunning performance.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup matplotlib style for consistent plots."""
        plt.style.use('default')
        sns.set_palette("Set2")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    # PITCHING ANALYSIS
    def analyze_pitching_performance(self, pitcher_name: str = None) -> Dict:
        """
        Comprehensive pitching analysis.
        
        Args:
            pitcher_name: Specific pitcher to analyze, or None for all pitchers
        """
        if pitcher_name:
            pitch_data = self.data[self.data['PitcherName'] == pitcher_name]
            if len(pitch_data) == 0:
                return {'error': f'No data found for pitcher {pitcher_name}'}
        else:
            pitch_data = self.data
        
        analysis = {
            'velocity_stats': {
                'mean': pitch_data['PitchVelo'].mean(),
                'std': pitch_data['PitchVelo'].std(),
                'min': pitch_data['PitchVelo'].min(),
                'max': pitch_data['PitchVelo'].max(),
                'median': pitch_data['PitchVelo'].median()
            },
            'pitch_outcomes': pitch_data['PitchOutcome'].value_counts().to_dict(),
            'total_pitches': len(pitch_data)
        }
        
        # Strike rate
        strikes = pitch_data['PitchOutcome'].isin(['Strike', 'Foul']).sum()
        analysis['strike_rate'] = strikes / len(pitch_data) * 100
        
        # Balls in play analysis
        bip_data = pitch_data[pitch_data['BallInPlay'] == True]
        if len(bip_data) > 0:
            analysis['balls_in_play'] = {
                'count': len(bip_data),
                'avg_exit_velo': bip_data['ExitVelo'].mean(),
                'outcomes': bip_data['Result'].value_counts().to_dict()
            }
        
        return analysis
    
    def plot_velocity_distribution(self, pitcher_name: str = None, save_path: str = None):
        """Plot pitch velocity distribution."""
        if pitcher_name:
            data = self.data[self.data['PitcherName'] == pitcher_name]
            title = f"Pitch Velocity Distribution - {pitcher_name}"
        else:
            data = self.data
            title = "Pitch Velocity Distribution - All Pitchers"
        
        plt.figure(figsize=(10, 6))
        plt.hist(data['PitchVelo'].dropna(), bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(data['PitchVelo'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {data["PitchVelo"].mean():.1f} mph')
        plt.xlabel('Pitch Velocity (mph)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # HITTING ANALYSIS
    def analyze_hitting_performance(self, batter_name: str = None) -> Dict:
        """
        Comprehensive hitting analysis.
        
        Args:
            batter_name: Specific batter to analyze, or None for all batters
        """
        if batter_name:
            hit_data = self.data[(self.data['BatterName'] == batter_name) & 
                               (self.data['BallInPlay'] == True)]
            if len(hit_data) == 0:
                return {'error': f'No hitting data found for batter {batter_name}'}
        else:
            hit_data = self.data[self.data['BallInPlay'] == True]
        
        analysis = {
            'total_balls_in_play': len(hit_data),
            'exit_velocity_stats': {
                'mean': hit_data['ExitVelo'].mean(),
                'std': hit_data['ExitVelo'].std(),
                'max': hit_data['ExitVelo'].max()
            },
            'launch_angle_stats': {
                'mean': hit_data['LaunchAng'].mean(),
                'std': hit_data['LaunchAng'].std()
            }
        }
        
        # Hit quality distribution
        if 'HitQuality' in hit_data.columns:
            analysis['hit_quality'] = hit_data['HitQuality'].value_counts().to_dict()
        
        # Sprint speed analysis
        sprint_data = hit_data[hit_data['BatterTimeToFirst'].notna()]
        if len(sprint_data) > 0:
            analysis['sprint_speed'] = {
                'avg_time_to_first': sprint_data['BatterTimeToFirst'].mean(),
                'best_time': sprint_data['BatterTimeToFirst'].min()
            }
        
        # Distance analysis
        distance_data = hit_data[hit_data['ActualDistance'].notna()]
        if len(distance_data) > 0:
            analysis['distance_stats'] = {
                'avg_distance': distance_data['ActualDistance'].mean(),
                'max_distance': distance_data['ActualDistance'].max()
            }
        
        return analysis
    
    def plot_exit_velocity_vs_launch_angle(self, save_path: str = None):
        """Create exit velocity vs launch angle scatter plot."""
        hit_data = self.data[(self.data['ExitVelo'].notna()) & 
                           (self.data['LaunchAng'].notna()) &
                           (self.data['BallInPlay'] == True)]
        
        plt.figure(figsize=(12, 8))
        
        # Color by hit outcome
        outcomes = hit_data['Result'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(outcomes)))
        
        for outcome, color in zip(outcomes, colors):
            outcome_data = hit_data[hit_data['Result'] == outcome]
            plt.scatter(outcome_data['LaunchAng'], outcome_data['ExitVelo'], 
                       alpha=0.6, label=outcome, s=50)
        
        # Add quality zones
        plt.axhspan(95, hit_data['ExitVelo'].max(), alpha=0.2, color='green', label='Hard Hit Zone')
        plt.axvspan(8, 32, alpha=0.2, color='red', label='Barrel Zone')
        
        plt.xlabel('Launch Angle (degrees)')
        plt.ylabel('Exit Velocity (mph)')
        plt.title('Exit Velocity vs Launch Angle')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # FIELDING ANALYSIS
    def analyze_fielding_performance(self, fielder_name: str = None) -> Dict:
        """
        Analyze fielding performance metrics.
        
        Args:
            fielder_name: Specific fielder to analyze, or None for all fielders
        """
        field_data = self.data[(self.data['EventPlayerName'].notna()) & 
                              (self.data['IsEventPlayer'] == True)]
        
        if fielder_name:
            field_data = field_data[field_data['EventPlayerName'] == fielder_name]
            if len(field_data) == 0:
                return {'error': f'No fielding data found for {fielder_name}'}
        
        analysis = {
            'total_fielding_plays': len(field_data),
            'route_efficiency': {
                'mean': field_data['FielderRouteEfficiency'].mean(),
                'std': field_data['FielderRouteEfficiency'].std()
            },
            'reaction_time': {
                'mean': field_data['FielderReaction'].mean(),
                'std': field_data['FielderReaction'].std()
            },
            'max_speed': {
                'mean': field_data['FielderMaxSpeed'].mean(),
                'max': field_data['FielderMaxSpeed'].max()
            }
        }
        
        # Success probability
        prob_data = field_data[field_data['FielderProbability'].notna()]
        if len(prob_data) > 0:
            analysis['catch_probability'] = {
                'mean': prob_data['FielderProbability'].mean(),
                'plays_above_50_percent': (prob_data['FielderProbability'] > 50).sum()
            }
        
        return analysis
    
    def plot_fielding_efficiency(self, save_path: str = None):
        """Plot fielding efficiency metrics."""
        field_data = self.data[(self.data['FielderRouteEfficiency'].notna()) & 
                              (self.data['FielderReaction'].notna())]
        
        if len(field_data) == 0:
            print("No fielding data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Route Efficiency Distribution
        axes[0,0].hist(field_data['FielderRouteEfficiency'], bins=15, alpha=0.7)
        axes[0,0].set_title('Route Efficiency Distribution')
        axes[0,0].set_xlabel('Route Efficiency (%)')
        axes[0,0].set_ylabel('Frequency')
        
        # Reaction Time Distribution
        axes[0,1].hist(field_data['FielderReaction'], bins=15, alpha=0.7, color='orange')
        axes[0,1].set_title('Reaction Time Distribution')
        axes[0,1].set_xlabel('Reaction Time (seconds)')
        axes[0,1].set_ylabel('Frequency')
        
        # Route Efficiency vs Reaction Time
        axes[1,0].scatter(field_data['FielderReaction'], field_data['FielderRouteEfficiency'], 
                         alpha=0.6)
        axes[1,0].set_title('Route Efficiency vs Reaction Time')
        axes[1,0].set_xlabel('Reaction Time (seconds)')
        axes[1,0].set_ylabel('Route Efficiency (%)')
        
        # Fielder Max Speed Distribution
        if field_data['FielderMaxSpeed'].notna().any():
            axes[1,1].hist(field_data['FielderMaxSpeed'].dropna(), bins=15, alpha=0.7, color='green')
            axes[1,1].set_title('Fielder Max Speed Distribution')
            axes[1,1].set_xlabel('Max Speed (mph)')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # BASERUNNING ANALYSIS
    def analyze_baserunning_performance(self) -> Dict:
        """Analyze baserunning metrics."""
        base_data = self.data[(self.data['BaserunnerMaxSpeed'].notna()) |
                             (self.data['BaserunnerInitial'].notna())]
        
        if len(base_data) == 0:
            return {'error': 'No baserunning data available'}
        
        analysis = {
            'total_baserunning_plays': len(base_data),
            'max_speed_stats': {
                'mean': base_data['BaserunnerMaxSpeed'].mean(),
                'max': base_data['BaserunnerMaxSpeed'].max(),
                'std': base_data['BaserunnerMaxSpeed'].std()
            }
        }
        
        # Base advancement analysis
        if 'BaseAdvancement' in base_data.columns:
            analysis['base_advancement'] = {
                'mean': base_data['BaseAdvancement'].mean(),
                'positive_advancement': (base_data['BaseAdvancement'] > 0).sum()
            }
        
        return analysis
    
    # TEAM PERFORMANCE COMPARISON
    def compare_team_performance(self) -> Dict:
        """Compare performance metrics between teams."""
        comparison = {}
        
        # Get unique pitchers (assuming they represent teams)
        pitchers = self.data['PitcherName'].unique()
        
        for pitcher in pitchers:
            pitcher_data = self.data[self.data['PitcherName'] == pitcher]
            
            comparison[pitcher] = {
                'pitches_thrown': len(pitcher_data),
                'avg_velocity': pitcher_data['PitchVelo'].mean(),
                'strike_rate': pitcher_data['PitchOutcome'].isin(['Strike', 'Foul']).sum() / len(pitcher_data) * 100
            }
            
            # Add hitting stats when this pitcher is batting
            hitting_data = pitcher_data[pitcher_data['BallInPlay'] == True]
            if len(hitting_data) > 0:
                comparison[pitcher]['avg_exit_velocity'] = hitting_data['ExitVelo'].mean()
        
        return comparison
    
    def generate_game_report(self, save_path: str = None) -> str:
        """Generate comprehensive game analysis report."""
        report = []
        report.append("=" * 60)
        report.append("GT BASEBALL 6TH TOOL - GAME ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic game info
        total_pitches = len(self.data)
        total_at_bats = self.data['AtBat'].nunique()
        innings_played = self.data['Inning'].max()
        
        report.append(f"GAME OVERVIEW:")
        report.append(f"  Total Pitches: {total_pitches}")
        report.append(f"  Total At-Bats: {total_at_bats}")
        report.append(f"  Innings Played: {innings_played}")
        report.append("")
        
        # Pitching summary
        pitch_analysis = self.analyze_pitching_performance()
        report.append("PITCHING SUMMARY:")
        report.append(f"  Average Velocity: {pitch_analysis['velocity_stats']['mean']:.1f} mph")
        report.append(f"  Velocity Range: {pitch_analysis['velocity_stats']['min']:.1f} - {pitch_analysis['velocity_stats']['max']:.1f} mph")
        report.append(f"  Strike Rate: {pitch_analysis['strike_rate']:.1f}%")
        report.append("")
        
        # Hitting summary
        hit_analysis = self.analyze_hitting_performance()
        if 'error' not in hit_analysis:
            report.append("HITTING SUMMARY:")
            report.append(f"  Balls in Play: {hit_analysis['total_balls_in_play']}")
            if 'exit_velocity_stats' in hit_analysis:
                report.append(f"  Average Exit Velocity: {hit_analysis['exit_velocity_stats']['mean']:.1f} mph")
                report.append(f"  Max Exit Velocity: {hit_analysis['exit_velocity_stats']['max']:.1f} mph")
            report.append("")
        
        # Top performers
        report.append("TOP PERFORMERS:")
        
        # Best exit velocity
        best_hit = self.data[self.data['ExitVelo'] == self.data['ExitVelo'].max()]
        if not best_hit.empty:
            row = best_hit.iloc[0]
            report.append(f"  Hardest Hit Ball: {row['BatterName']} - {row['ExitVelo']:.1f} mph")
        
        # Fastest pitch
        fastest_pitch = self.data[self.data['PitchVelo'] == self.data['PitchVelo'].max()]
        if not fastest_pitch.empty:
            row = fastest_pitch.iloc[0]
            report.append(f"  Fastest Pitch: {row['PitcherName']} - {row['PitchVelo']:.1f} mph")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text

# Example usage
if __name__ == "__main__":
    # This would be used with the data loader
    from data_loader import GTBaseballDataLoader
    
    loader = GTBaseballDataLoader()
    game_data = loader.load_game_data("gt_vs_louisville_game1.csv")
    
    analyzer = GTBaseballAnalyzer(game_data)
    
    # Generate comprehensive report
    report = analyzer.generate_game_report("game_analysis_report.txt")
    print(report)
    
    # Create visualizations
    analyzer.plot_velocity_distribution()
    analyzer.plot_exit_velocity_vs_launch_angle()
    analyzer.plot_fielding_efficiency()