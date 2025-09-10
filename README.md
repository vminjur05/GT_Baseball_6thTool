# GT_Baseball_6thTool

A baseball analytics tool for Georgia Tech Baseball.

## Setup Instructions

### Prerequisites
- macOS with Python 3.11+ installed
- Terminal access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GT_Baseball_6thTool

2. **Create a virtual environment**
   Your terminal prompt should now show (venv) at the beginning.
   ```bash
    python3 -m venv venv
    source venv/bin/activate
   
4. **Install dependencies**

    Option A: Use updated package versions (Recommended)

    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install pandas matplotlib numpy

    Option B: If you need exact versions from requirements.txt

    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install -r requirements.txt

## Troubleshooting
    If you get "command not found: python":

    - Use python3 instead of python
    - Use pip3 instead of pip
    - If you get "externally-managed-environment" error:

## Extra Notes
    Make sure you've activated your virtual environment first (run this each time you start working):

    source venv/bin/activate

    Deactivate environment (when you're done):

    deactivate
