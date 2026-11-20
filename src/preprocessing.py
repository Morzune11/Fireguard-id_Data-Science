"""
Preprocessing & Spatial Feature Engineering Module for FIREGUARD ID.
"""
import numpy as np
import pandas as pd
from matplotlib.path import Path
from sklearn.preprocessing import StandardScaler

def clip_hotspots_to_polygon(df_hotspots, boundary_geojson):
    """Performs exact polygon point-in-polygon clipping."""
    geom = boundary_geojson["features"][0]["geometry"]
    polygons = []
    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            polygons.append(Path(ring))
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                polygons.append(Path(ring))

    points = df_hotspots[["longitude", "latitude"]].values
    inside_mask = np.zeros(len(points), dtype=bool)
    for p in polygons:
        inside_mask |= p.contains_points(points)

    return df_hotspots[inside_mask].copy()

def fit_transform_scaler(train_features, val_features=None, test_features=None):
    """Fits StandardScaler ONLY on training features to prevent leakage."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features) if val_features is not None else None
    test_scaled = scaler.transform(test_features) if test_features is not None else None
    return scaler, train_scaled, val_scaled, test_scaled
