import geopandas as gpd
from pathlib import Path
from typing import Optional


def save_gdf(gdf: gpd.GeoDataFrame, output_path: Path, filename: Optional[str] = "polygon.shp") -> Path:

    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / filename
    gdf.to_file(filename=output_file, driver="ESRI Shapefile")
    return output_file