#!/usr/bin/env python3
"""
Quick column inspector for GT Baseball data files.
Helps identify column naming differences between CSV and Parquet files.
"""

import pandas as pd
import sys
from pathlib import Path

def inspect_file(filepath):
    """Inspect columns in CSV or Parquet file."""
    path = Path(filepath)
    
    try:
        # Read file based on extension
        if path.suffix.lower() in ['.parquet', '.parq']:
            df = pd.read_parquet(filepath)
            file_type = "Parquet"
        else:
            df = pd.read_csv(filepath)
            file_type = "CSV"
        
        print(f"\n{'='*70}")
        print(f"File: {path.name}")
        print(f"Type: {file_type}")
        print(f"{'='*70}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        print(f"\n{'Column Name':<40} {'Data Type':<15} {'Non-Null':<10} {'Sample'}")
        print('-'*100)
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            
            # Get a sample value
            sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
            if isinstance(sample_val, (int, float)):
                sample = f"{sample_val:.2f}" if isinstance(sample_val, float) else str(sample_val)
            else:
                sample = str(sample_val)[:30]
            
            print(f"{col:<40} {dtype:<15} {non_null:<10} {sample}")
        
        # Check for baserunning-related columns
        print(f"\n{'BASERUNNING COLUMNS':^100}")
        print('-'*100)
        baserunning_keywords = ['baserunner', 'runner', 'speed', 'base', 'lead']
        baserunning_cols = [col for col in df.columns 
                           if any(kw in col.lower() for kw in baserunning_keywords)]
        
        if baserunning_cols:
            print("Found baserunning-related columns:")
            for col in baserunning_cols:
                print(f"  • {col}")
        else:
            print("⚠️  No baserunning columns found!")
        
        # Check for fielding columns
        print(f"\n{'FIELDING COLUMNS':^100}")
        print('-'*100)
        fielding_keywords = ['fielder', 'route', 'reaction', 'probability', 'catch']
        fielding_cols = [col for col in df.columns 
                        if any(kw in col.lower() for kw in fielding_keywords)]
        
        if fielding_cols:
            print("Found fielding-related columns:")
            for col in fielding_cols:
                print(f"  • {col}")
        else:
            print("⚠️  No fielding columns found!")
        
        # Check for hitting columns
        print(f"\n{'HITTING COLUMNS':^100}")
        print('-'*100)
        hitting_keywords = ['exit', 'launch', 'velo', 'velocity', 'distance', 'batter']
        hitting_cols = [col for col in df.columns 
                       if any(kw in col.lower() for kw in hitting_keywords)]
        
        if hitting_cols:
            print("Found hitting-related columns:")
            for col in hitting_cols:
                print(f"  • {col}")
        else:
            print("⚠️  No hitting columns found!")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return

def compare_files(file1, file2):
    """Compare columns between two files."""
    try:
        # Read both files
        df1 = pd.read_parquet(file1) if file1.endswith('.parquet') else pd.read_csv(file1)
        df2 = pd.read_parquet(file2) if file2.endswith('.parquet') else pd.read_csv(file2)
        
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        print(f"\n{'='*70}")
        print("COLUMN COMPARISON")
        print(f"{'='*70}")
        
        print(f"\n📁 File 1: {Path(file1).name} ({len(cols1)} columns)")
        print(f"📁 File 2: {Path(file2).name} ({len(cols2)} columns)")
        
        # Common columns
        common = cols1 & cols2
        print(f"\n✅ Common columns ({len(common)}):")
        for col in sorted(common):
            print(f"  • {col}")
        
        # Only in file 1
        only1 = cols1 - cols2
        if only1:
            print(f"\n⚠️  Only in {Path(file1).name} ({len(only1)}):")
            for col in sorted(only1):
                print(f"  • {col}")
        
        # Only in file 2
        only2 = cols2 - cols1
        if only2:
            print(f"\n⚠️  Only in {Path(file2).name} ({len(only2)}):")
            for col in sorted(only2):
                print(f"  • {col}")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Error comparing files: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Inspect single file:  python inspect_columns.py <file>")
        print("  Compare two files:    python inspect_columns.py <file1> <file2>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        inspect_file(sys.argv[1])
    elif len(sys.argv) == 3:
        inspect_file(sys.argv[1])
        inspect_file(sys.argv[2])
        compare_files(sys.argv[1], sys.argv[2])