"""
Requirements Installation Utility
This module provides functionality to install project dependencies from requirements.txt.
It is used during development setup and deployment to ensure all required packages are installed.
"""

import subprocess
import sys

def install_requirements():
    """
    Installs all project dependencies listed in requirements.txt using pip.
    
    Raises:
        subprocess.CalledProcessError: If pip install fails
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])