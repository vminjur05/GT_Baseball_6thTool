import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GTBaseballDataLoader:
    """
    Handles loading and processing GT Baseball CSV data files.
    Designed for the GT Baseball 6th Tool project data format.
    """
    
    def __init__(self, data_directory: str = "data"):
        self.data_dir = data_directory
        self.game_data = None
        
        # Column definitions based on your CSV structure
        self.required_columns = [
            'Inning', 'AtBat', 'PitcherName', 'BatterName', 'Result', 'PitchVelo'
        ]
        
        self.optional_columns = [
            'BatterTimeToFirst', 'BatterTop', 'ExitVelo', 'LaunchAng', 'ActualDistance',
            'BaserunnerMaxSpeed', 'BaserunnerInitial', 'BaserunnerSecondary', 'BaserunnerFinal',
            'IsEventPlayer', 'EventPlayerName', 'FielderProbability', 'FielderRouteEfficiency',
            'FielderMove', 'FielderReaction', 'FielderReactionAngle', 'FielderTransfer',
            'FielderThrow', 'FielderThrowDistance', 'FielderMaxSpeed'
        ]
    
    def load_game_data(self, filepath: str, game_id: str = None) -> pd.DataFrame:
        """
        Load GT Baseball game data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            game_id: Optional game identifier for tracking
            
        Returns:
            Processed DataFrame with clean data types
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Load CSV
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            
            # Add game_id if provided
            if game_id:
                df['GameID'] = game_id
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and process data
            df = self._clean_data(df)
            df = self._add_derived_columns(df)
            
            logger.info(f"Processed data: {len(df)} total pitches, {df['AtBat'].nunique()} at-bats")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types."""
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_cols = [
            'Inning', 'AtBat', 'PitchVelo', 'BatterTimeToFirst', 'BatterTop',
            'ExitVelo', 'LaunchAng', 'ActualDistance', 'BaserunnerMaxSpeed',
            'BaserunnerInitial', 'BaserunnerSecondary', 'BaserunnerFinal',
            'FielderProbability', 'FielderRouteEfficiency', 'FielderMove',
            'FielderReaction', 'FielderReactionAngle', 'FielderTransfer',
            'FielderThrow', 'FielderThrowDistance', 'FielderMaxSpeed'
        ]
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Convert boolean columns
        if 'IsEventPlayer' in df_clean.columns:
            df_clean['IsEventPlayer'] = df_clean['IsEventPlayer'].astype(bool)
        
        # Clean string columns
        string_cols = ['PitcherName', 'BatterName', 'Result', 'EventPlayerName']
        for col in string_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        return df_clean
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated columns for analysis."""
        df_derived = df.copy()
        
        # Pitch outcome categories
        df_derived['PitchOutcome'] = df_derived['Result'].apply(self._categorize_pitch_outcome)
        
        # Ball in play indicator
        df_derived['BallInPlay'] = df_derived['Result'].str.contains('In Play', na=False)
        
        # Hit quality (based on exit velocity and launch angle)
        if 'ExitVelo' in df_derived.columns and 'LaunchAng' in df_derived.columns:
            df_derived['HitQuality'] = df_derived.apply(self._calculate_hit_quality, axis=1)
        
        # Base running efficiency
        if all(col in df_derived.columns for col in ['BaserunnerInitial', 'BaserunnerFinal']):
            df_derived['BaseAdvancement'] = df_derived['BaserunnerFinal'] - df_derived['BaserunnerInitial']
        
        # Fielding efficiency score
        if 'FielderRouteEfficiency' in df_derived.columns and 'FielderReaction' in df_derived.columns:
            df_derived['FieldingEfficiency'] = df_derived.apply(self._calculate_fielding_efficiency, axis=1)
        
        return df_derived
    
    def _categorize_pitch_outcome(self, result: str) -> str:
        """Categorize pitch outcomes into main types."""
        if pd.isna(result):
            return 'Unknown'
        
        result = str(result).lower()
        
        if 'ball' in result:
            return 'Ball'
        elif 'strike' in result:
            return 'Strike'
        elif 'in play' in result:
            return 'In Play'
        elif 'foul' in result:
            return 'Foul'
        elif 'hit by pitch' in result or 'hitbypitch' in result:
            return 'HBP'
        else:
            return 'Other'
    
    def _calculate_hit_quality(self, row) -> str:
        """Calculate hit quality based on exit velo and launch angle."""
        if pd.isna(row['ExitVelo']) or pd.isna(row['LaunchAng']):
            return 'Unknown'
        
        exit_velo = row['ExitVelo']
        launch_angle = row['LaunchAng']
        
        # Baseball analytics standard classifications
        if exit_velo >= 95 and 8 <= launch_angle <= 32:
            return 'Barrel'
        elif exit_velo >= 95:
            return 'Hard Hit'
        elif exit_velo < 60:
            return 'Weak Contact'
        else:
            return 'Medium Contact'
    
    def _calculate_fielding_efficiency(self, row) -> float:
        """Calculate fielding efficiency score (0-100)."""
        if pd.isna(row['FielderRouteEfficiency']) or pd.isna(row['FielderReaction']):
            return np.nan
        
        # Weight route efficiency more heavily than reaction time
        route_score = row['FielderRouteEfficiency']
        reaction_score = 100 - (row['FielderReaction'] * 10)  # Convert reaction time to score
        reaction_score = max(0, min(100, reaction_score))  # Cap between 0-100
        
        return (route_score * 0.7) + (reaction_score * 0.3)
    
    def get_pitch_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for pitches."""
        if df is None or len(df) == 0:
            return {}
        
        summary = {
            'total_pitches': len(df),
            'total_at_bats': df['AtBat'].nunique(),
            'avg_pitch_velocity': df['PitchVelo'].mean(),
            'pitch_outcomes': df['PitchOutcome'].value_counts().to_dict(),
            'pitchers': df['PitcherName'].unique().tolist(),
            'batters': df['BatterName'].unique().tolist()
        }
        
        # Add batting statistics if available
        balls_in_play = df[df['BallInPlay'] == True]
        if len(balls_in_play) > 0:
            summary['balls_in_play'] = len(balls_in_play)
            if 'ExitVelo' in df.columns:
                summary['avg_exit_velocity'] = balls_in_play['ExitVelo'].mean()
            if 'LaunchAng' in df.columns:
                summary['avg_launch_angle'] = balls_in_play['LaunchAng'].mean()
        
        return summary
    
    def get_player_stats(self, df: pd.DataFrame, player_type: str = 'both') -> Dict:
        """
        Get player statistics.
        
        Args:
            df: Game data DataFrame
            player_type: 'pitcher', 'batter', or 'both'
        """
        stats = {}
        
        if player_type in ['pitcher', 'both']:
            pitcher_stats = df.groupby('PitcherName').agg({
                'PitchVelo': ['mean', 'std', 'count'],
                'PitchOutcome': lambda x: (x == 'Strike').sum(),
                'BallInPlay': 'sum'
            }).round(2)
            
            pitcher_stats.columns = ['avg_velocity', 'velocity_std', 'total_pitches', 'strikes', 'balls_in_play']
            stats['pitchers'] = pitcher_stats.to_dict('index')
        
        if player_type in ['batter', 'both']:
            balls_in_play = df[df['BallInPlay'] == True]
            if len(balls_in_play) > 0:
                batter_stats = balls_in_play.groupby('BatterName').agg({
                    'ExitVelo': 'mean',
                    'LaunchAng': 'mean',
                    'ActualDistance': 'mean',
                    'BatterTimeToFirst': 'mean'
                }).round(2)
                
                batter_stats.columns = ['avg_exit_velocity', 'avg_launch_angle', 'avg_distance', 'avg_sprint_speed']
                stats['batters'] = batter_stats.to_dict('index')
        
        return stats
    
    def load_multiple_games(self, file_patterns: List[str]) -> pd.DataFrame:
        """Load and combine multiple game files."""
        all_games = []
        
        for i, pattern in enumerate(file_patterns):
            if os.path.exists(pattern):
                game_df = self.load_game_data(pattern, game_id=f"Game_{i+1}")
                all_games.append(game_df)
        
        if all_games:
            combined_df = pd.concat(all_games, ignore_index=True)
            logger.info(f"Combined {len(all_games)} games with {len(combined_df)} total pitches")
            return combined_df
        else:
            logger.warning("No game files found")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    loader = GTBaseballDataLoader("data")
    
    # Load a single game
    game_data = loader.load_game_data("gt_vs_louisville_game1.csv", "GT_vs_LOU_1")
    
    # Get summary statistics
    summary = loader.get_pitch_summary(game_data)
    print("Game Summary:", summary)
    
    # Get player statistics
    player_stats = loader.get_player_stats(game_data)
    print("Player Stats:", player_stats)