"""
Utilities & Helper Functions for FIREGUARD ID.
"""
import os

def ensure_dir(dir_path):
    """Ensures a directory exists."""
    os.makedirs(dir_path, exist_ok=True)
