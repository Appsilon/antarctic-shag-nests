from __future__ import annotations
import argparse
from typing import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from utils.bounding_box import BoundingBox, get_bounding_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "BBox merger",
        description="Right now different dates are from files like 'patch28_19_11_26' are found by looking for the first `_` character and taking the characters after `_` as the _big tif_ identifier.",
    )
    parser.add_argument(
        "--predictions",
        help="Shapefile with predictions, output from converter script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--nest_size_threshold",
        help="Minimum area of a single nest",
        type=float,
        default=0.5 * 0.5**2,
    )
    parser.add_argument(
        "--iou_threshold",
        help="IoU above which we remove overlapping bboxes",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--min_side_length", help="Minimum nest side length in meters", type=float, default=0.35
    )
    parser.add_argument(
        "--max_side_length", help="Maximum nest side length in meters", type=float, default=0.65
    )
    parser.add_argument(
        "--min_sides_ratio",
        help="Squary parameter, Minimum ratio of min(a,b) / max(a,b). 1 for perfect square, 0 for perfect segment",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()
    if not args.predictions.endswith("shp"):
        raise ValueError("Expecting predictions in shapefile")

    return args


def remove_redundant(bboxes: Sequence[BoundingBox], iou_threshold: float) -> list[BoundingBox]:
    bboxes = sorted(bboxes, key=lambda b: b.score, reverse=True)
    k = 0
    while k < len(bboxes):
        b0 = bboxes[k]
        to_remove: list[int] = []
        for i, b in enumerate(bboxes[k + 1 :]):
            if b0.iou(b) > iou_threshold:
                to_remove.append(k + i + 1)
        for i in to_remove[::-1]:  # we have to remove items from the end as we don't do it at once
            bboxes.pop(i)
        k += 1
    return bboxes


def process_groups(df: gpd.GeoDataFrame, iou_threshold: float) -> gpd.GeoDataFrame:
    
    if "BoundingBox" not in df.columns:
        df = get_bounding_box(df)
    
    columns = ["image_id", "coords", "score", "geometry"]
    output_rows = []
    df["date_id"] = df["image_id"].apply(lambda s: s[s.find("_") + 1 :])
    for file_name, group in df.groupby("date_id"):
        bboxes = remove_redundant(group["BoundingBox"].to_list(), iou_threshold)

        for b in bboxes:
            lat_point_list = [b.xmin, b.xmax, b.xmax, b.xmin, b.xmin]
            lon_point_list = [b.ymin, b.ymin, b.ymax, b.ymax, b.ymin]
            poly = Polygon(zip(lat_point_list, lon_point_list))

            output_rows.append(
                (
                    file_name,
                    *b.to_tuple(),
                    poly,
                )
            )
    crs = "epsg:32721"
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(output_rows, columns=columns), crs=crs, geometry="geometry"
    )
    return gdf


def filter_boxes(
    df: gpd.GeoDataFrame, 
    min_area: float = 0., 
    max_area: float = float("inf"),
    min_side: float = 0.,
    max_side: float = float("inf"),
    min_aspect_ratio: float = 0) -> gpd.GeoDataFrame:

    if "BoundingBox" not in df.columns:
        df = get_bounding_box(df)

    # Filter on nest size
    df = df.loc[df["BoundingBox"].apply(lambda b: b.area()) >= min_area]
    df = df.loc[df["BoundingBox"].apply(lambda b: b.area()) <= max_area]
    # Filter on sides lengths
    df = df.loc[df["BoundingBox"].apply(lambda b: min(b.w, b.h)) >= min_side]
    df = df.loc[df["BoundingBox"].apply(lambda b: max(b.w, b.h)) <= max_side]
    # Filter on squeryness
    df = df.loc[df["BoundingBox"].apply(lambda b: min(b.w, b.h) / max(b.w, b.h)) >= min_aspect_ratio]

    return df


def main(predictions: gpd.GeoDataFrame, min_area: float, min_side: float, max_side: float, min_aspect_ratio: float, iou_threshold: float) -> None:

    predictions = get_bounding_box(predictions)
    predictions = filter_boxes(
        predictions, 
        min_area=min_area, 
        min_side=min_side, 
        max_side=max_side, 
        min_aspect_ratio=min_aspect_ratio)

    # Merge predictions
    predictions = process_groups(predictions, iou_threshold)
    return predictions


if __name__ == "__main__":
    args = parse_args()
    gdf = gpd.read_file(args.predictions)
    main(
        predictions=gdf,
        min_area=args.nest_size_threshold, 
        min_side=args.min_side_length, 
        max_side=args.max_side_length, 
        min_aspect_ratio=args.min_sides_ratio,
        iou_threshold=args.iou_threshold
    )
