from __future__ import annotations

import geopandas as gpd
from dataclasses import dataclass

coords = str


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float

    @property
    def h(self):
        return self.ymax - self.ymin

    @property
    def w(self):
        return self.xmax - self.xmin

    def iou(self, b: BoundingBox) -> float:
        """Based on https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles"""
        dx = min(self.xmax, b.xmax) - max(self.xmin, b.xmin)
        dy = min(self.ymax, b.ymax) - max(self.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            I = dx * dy
            return I / (self.area() + b.area() - I)
        else:
            return 0.0

    def area(self) -> float:
        return self.w * self.h

    @classmethod
    def from_bbox_string(cls, bbox: str, score: float):
        """Creates BBox from string like
        '432479.2021,3104564.805501156,432479.2519121387,3104565.279301734'"""
        xmin, ymin, xmax, ymax = [float(e) for e in bbox.split(",")]
        return cls(xmin, ymin, xmax, ymax, score)

    def to_tuple(self) -> tuple[coords, float]:
        return f"{self.xmin},{self.ymin},{self.xmax},{self.ymax}", self.score


def get_bounding_box(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    df["BoundingBox"] = df.apply(
        lambda r: BoundingBox.from_bbox_string(r["coords"], r["score"]), axis=1
    )

    return df


def coord2pix(cx, cy, top_left, dx, dy):
    # takes x and y geographical coordinate
    # returns row, col for Array[row, col]
    row = int((top_left[1] - cy) / dy)
    col = int((cx - top_left[0]) / dx)
    return row, col


def bbox_from_image_geometry(geom, top_left: list[float, float], pixel_size: list[float, float]):

    # In geographical coordinates we have x and y coordinate
    # to describe horizontal and vertical position.
    # In python array coordinates go like: Array[row, col],
    # which means that X and Y axis are switched.
    # Additionally in Qgis xMinimum, yMinimum refer to the
    # left bottom corner and xMaximum, yMaximum refer to the
    # right upper corner of the rectangle. In Array we always
    # start counting from the very top row which means that
    # vertical axis is inverted.

    row_max, col_min = coord2pix(
        geom.xMinimum(),
        geom.yMinimum(),
        top_left=top_left,
        dx=pixel_size[0],
        dy=pixel_size[1],
    )

    row_min, col_max = coord2pix(
        geom.xMaximum(),
        geom.yMaximum(),
        top_left=top_left,
        dx=pixel_size[0],
        dy=pixel_size[1],
    )

    width = col_max - col_min
    height = row_max - row_min
    x_centre = col_min + (width) / 2
    y_centre = row_min + (height) / 2

    return x_centre, y_centre, width, height