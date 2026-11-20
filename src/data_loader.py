"""
Data Loader Module for FIREGUARD ID.
Loads raw satellite hotspots, administrative boundary, and vulnerability grid datasets.
"""
import os
import json
import pandas as pd

def load_boundary(geojson_path="data/processed/riau_boundary.geojson"):
    """Loads Riau GeoJSON boundary."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_hotspots(csv_path="data/processed/riau_viirs_hotspots_2021_2025.csv"):
    """Loads cleaned VIIRS hotspots within Riau."""
    return pd.read_csv(csv_path)

def load_grid_profiles(csv_path="data/processed/riau_vulnerability_grid_01deg.csv"):
    """Loads 0.1 degree grid spatial profiles."""
    return pd.read_csv(csv_path)

def load_splits(split_dir="data/split"):
    """Loads train, validation, and test sets."""
    train_df = pd.read_csv(os.path.join(split_dir, "train_scaled.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_scaled.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_scaled.csv"))
    return train_df, val_df, test_df
