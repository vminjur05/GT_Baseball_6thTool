#!/usr/bin/env python3
"""
GT Baseball 6th Tool - Setup and Quick Start Script
This script helps you get started with your baseball analytics quickly.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def setup_project_structure():
    """Create the project directory structure."""
    directories = [
        'csv_data',        # Directory for CSV files
        'scripts',          # Directory for Python scripts
        'scripts/visualizations',  # Directory for visualizations (new)
        'scripts/reports',         # Directory for reports (new)
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n📁 Project structure created successfully!")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'streamlit', 'scipy', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# GT Baseball 6th Tool Configuration

# Data Settings
DATA_DIR = "csv_data"          # Path to raw data
PROCESSED_DIR = "scripts/processed"   # Path to processed data
REPORTS_DIR = "scripts/reports"      # Path to reports

# Analysis Settings
DEFAULT_VELOCITY_THRESHOLD = 95  # mph for "hard hit" classification
BARREL_LAUNCH_ANGLE_MIN = 8     # degrees
BARREL_LAUNCH_ANGLE_MAX = 32    # degrees

# Visualization Settings
PLOT_STYLE = "gt_colors"
FIGURE_SIZE = (12, 8)
DPI = 300

# Team Colors (GT)
GT_GOLD = "#B3A369" 
GT_NAVY = "#003057"
GT_WHITE = "#FFFFFF"
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Created config.py")

def create_requirements_txt():
    """Create requirements.txt file."""
    requirements = """pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.28.0
scipy>=1.9.0
scikit-learn>=1.1.0
jupyter>=1.0.0
openpyxl>=3.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_quick_start_notebook():
    """Create a Jupyter notebook for quick start."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GT Baseball 6th Tool - Quick Start\n",
                    "\n",
                    "This notebook will help you get started with analyzing your GT Baseball data.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import required libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Import our custom modules\n",
                    "from scripts.data_loader import GTBaseballDataLoader\n",
                    "from scripts.baseball_analyzer import GTBaseballAnalyzer\n",
                    "\n",
                    "print('✅ All imports successful!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load your data\n",
                    "loader = GTBaseballDataLoader('csv_data')\n",
                    "\n",
                    "# Replace 'your_file.csv' with your actual CSV file name\n",
                    "game_data = loader.load_game_data('your_file.csv', 'Game_1')\n",
                    "\n",
                    "print(f'Loaded {len(game_data)} pitches')\n",
                    "game_data.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create analyzer and generate quick summary\n",
                    "analyzer = GTBaseballAnalyzer(game_data)\n",
                    "\n",
                    "# Generate comprehensive report\n",
                    "report = analyzer.generate_game_report()\n",
                    "print(report)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create some basic visualizations\n",
                    "analyzer.plot_velocity_distribution()\n",
                    "analyzer.plot_exit_velocity_vs_launch_angle()\n",
                    "analyzer.plot_fielding_efficiency()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open('notebooks/GT_Baseball_Quick_Start.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Created Jupyter notebook: notebooks/GT_Baseball_Quick_Start.ipynb")

def create_run_dashboard_script():
    """Create a script to easily run the Streamlit dashboard."""
    script_content = """#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("🚀 Starting GT Baseball 6th Tool Dashboard...")
    
    # Check if we're in the right directory
    if not os.path.exists('scripts/baseball_dashboard.py'):
        print("❌ baseball_dashboard.py not found!")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'scripts/baseball_dashboard.py'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\\n👋 Dashboard stopped.")

if __name__ == "__main__":
    main()
"""
    
    with open('run_dashboard.py', 'w') as f:
        f.write(script_content)
    
    # Make it executable on Unix systems
    try:
        os.chmod('run_dashboard.py', 0o755)
    except:
        pass
    
    print("✅ Created run_dashboard.py")