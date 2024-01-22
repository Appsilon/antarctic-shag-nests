import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon

from preprocessing.dataset import Dataset


def main(predictions: gpd.GeoDataFrame, dataset: Dataset, name: str, *args, **kwargs) -> (gpd.GeoDataFrame, pd.DataFrame):
    """
    Process prediction results for a given dataset
    dataset: dataset from metadata.yaml
    predictions: GeoDataFrame of predictions
    kw/args: passed to metrics function
    """

    results = []
    for img, shp in zip(dataset.images, dataset.shapefiles):

        image_id = Path(img).stem
        image_preds = predictions[predictions["image_id"].str.contains(image_id)]

        # Load ground truth
        gt = dataset.load_shp_data(shp)
        gt.geometry = points2rectangles(gt.geometry)
        
        # Process results
        image_preds = process(image_preds, gt)

        # Edge case where no predictions made
        if not image_preds.shape[0]:
            image_preds.loc[0, "image_id"] = image_id

        # Save metadata
        image_preds["location"] = name
        image_preds["num_truth"] = gt.shape[0]

        results.append(image_preds)

    result_df = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=results[0].crs)
    result_metrics = metrics(result_df, *args, **kwargs)

    return result_df, result_metrics


def process(
    preds: gpd.geodataframe.GeoDataFrame,
    nests_df: gpd.geodataframe.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Processes predictions with matching nest and IoU.
    Predictions are sorted by confidence and only one prediction is allowed per nest,
    therefore predictions of the same nest with lower confidence will show as false positive with 0 IoU.
    """

    # clean nests geodata
    nests_df = nests_df[~nests_df["geometry"].isna()]
    nests_df = nests_df.drop_duplicates(subset=["geometry"], keep="first")
    nests_df.reset_index(drop=True, inplace=True)

    preds = preds.copy()
    preds.sort_values("score", ascending=False, inplace=True)
    preds.reset_index(drop=True, inplace=True)

    preds_nest = np.zeros(preds.shape[0]) - 1
    preds_max_iou = np.zeros(preds.shape[0])

    for pred_i, pred_row in preds.iterrows():
        
        for nest_i, nest_row in nests_df.iterrows():
            
            if nest_i in preds_nest:
                continue

            nest_iou = iou(pred_row["geometry"], nest_row["geometry"])

            if nest_iou > preds_max_iou[pred_i]:

                # TODO: Nest ID may be irrelevant here since we are resetting index above. 
                #   Could remove or use original index (i.e. don't drop above) if needed.
                preds_nest[pred_i] = nest_i
                preds_max_iou[pred_i] = nest_iou

    preds = preds.assign(nest_id=preds_nest.astype(int), iou=preds_max_iou)

    return preds


def point2polygon(x: float, y: float, h: float = 0.5, w: float = 0.5) -> Polygon:
    p1 = (x - w / 2, y - h / 2)
    p2 = (x - w / 2, y + h / 2)
    p3 = (x + w / 2, y + h / 2)
    p4 = (x + w / 2, y - h / 2)
    return Polygon([p1, p2, p3, p4, p1])


def points2rectangles(geometry: gpd.GeoSeries) -> gpd.GeoSeries:
    return geometry.apply(lambda p: point2polygon(p.x, p.y))


def iou(p1: Polygon, p2: Polygon) -> float:
    intersection = p1.intersection(p2)
    return intersection.area / (p1.area + p2.area - intersection.area)


def mean_average_precision(pred_df: gpd.geodataframe.GeoDataFrame, n_pos: int, start: float = 0.5, end: float = 0.95, step: float = 0.05):

    scores = []
    for threshold in np.linspace(start, end, int((end - start)/step + 1)):

        scores.append(average_precision(pred_df, n_pos, threshold))

    return np.mean(scores)


def average_precision(pred_df: gpd.geodataframe.GeoDataFrame, n_pos: int, iou_threshold: float = 0.5):
    """
    Calculating average precision using the area under curve approach and "stepped" or "flatened" precision
    i.e. max precision to the right of the precision recall graph
    """

    pred_df = pred_df.sort_values("score", ascending=False).copy()
    pred_df.reset_index(drop=True, inplace=True)
    pred_df["tp"] = pred_df["iou"] >= iou_threshold

    pred_df["precision"] = pred_df["tp"].cumsum() / (pred_df.index + 1)
    pred_df["recall"] = pred_df["tp"].cumsum() / n_pos

    recall_diff = pred_df["recall"].diff()

    area = 0
    for i in range(recall_diff.shape[0] - 1):
        # Get max precision to the right
        max_p = pred_df.loc[i:, "precision"].max()
        # Calculate area
        area += max_p * recall_diff[i + 1]

    return area


def metrics(
    preds: gpd.geodataframe.GeoDataFrame,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Calculates overall metrics from processed predictions
    """

    all_metrics = []    
    for group_key, group in preds.groupby(["location", "image_id"]):

        num_truth = group["num_truth"].iloc[0]
        
        group = group[(group.score >= score_threshold)].copy()
        group["fp"] = (group["nest_id"] == -1) | (group.iou < iou_threshold)

        fp = group["fp"].sum()
        tp = (~group["fp"]).sum()
        fn = num_truth - tp

        all_metrics.append({
            "location": group_key[0], 
            "image_id": group_key[1],
            "n_preds": group.shape[0],
            "n_nests": num_truth,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "f1_score": tp/(tp + (fp + fn)/2),
            "CAS": 1 - np.abs(tp + fp - num_truth)/num_truth,
            "mAP[0.5]": average_precision(group, n_pos=num_truth, iou_threshold=0.5),
            "mAP[0.75]": average_precision(group, n_pos=num_truth, iou_threshold=0.75),
            "mAP[0.5:0.95]": mean_average_precision(group, n_pos=num_truth, start=0.5, end=0.95, step=0.05),
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold
        })

    metrics = pd.DataFrame(all_metrics)

    return metrics