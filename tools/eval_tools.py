import os
import pathlib
import configparser

from typing import Dict, List, Optional, Union

import motmetrics as mm
import pandas as pd


GT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
]

EV_COLUMNS = [
    "FrameId",
    "Id",
    "X",
    "Y",
    "Width",
    "Height",
    "Xworld",
    "Yworld",
]


def evaluate_multi_scene(prediction_dfs, ground_truth_dfs, names=None):
    """Takes prediction and ground truth dataframes and runs motmetrics evaluation
    on a multiple scenes. For evaluation of multi-camera scenes, first combine a
    list of single-camera predictions and ground truths using `combine_dataframes`
    Args:
        prediction_dfs (_type_): _description_
        ground_truth_dfs (_type_): _description_
        names (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """
    if names is None:
        names = ["Untitled %s" % (i + 1) for i in range(len(prediction_dfs))]
    ground_truths = {
        name: ground_truth_df for name, ground_truth_df in zip(names, ground_truth_dfs)
    }
    predictions = {
        name: prediction_df for name, prediction_df in zip(names, prediction_dfs)
    }
    accs, names = evaluate(ground_truths, predictions)
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        names=names,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
    return summary


def evaluate_single_scene(prediction_df, ground_truth_df, name=None) -> pd.DataFrame:
    """Takes a prediction and ground truth dataframe and runs motmetrics evaluation
    on a single scene. For evaluation of multi-camera scenes, first combine a list
    of single-camera predictions and ground truths using `combine_dataframes`.
    Args:
        prediction_df (_type_): Multi-camera predictions.
        ground_truth_df (_type_): Multi-camera ground truth.
        name (str): Scene name. Defaults to None.
    """
    return evaluate_multi_scene([prediction_df], [ground_truth_df], [name])


def mot_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a MOT-style dataframe (with named columns [frame, id, ...])
    and converts it to a dataframe with column names required by motmetrics.
    Args:
        df (pd.DataFrame): Input MOT-style dataframe.
    Returns:
        pd.DataFrame: Output dataframe ready to use in motmetrics evaluation.
    """
    _df = df.rename(
        columns={
            "frame": "FrameId",
            "id": "Id",
            "bb_left": "X",
            "bb_top": "Y",
            "bb_width": "Width",
            "bb_height": "Height",
            "conf": "Confidence",
        }
    )
    columns_to_int = ["FrameId", "Id", "X", "Y", "Width", "Height"]
    columns_to_float = ["Confidence"]
    _df[columns_to_int] = _df[columns_to_int].astype(int)
    _df[columns_to_float] = _df[columns_to_float].astype(float)
    return _df


def read_txt(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    _df = pd.read_csv(path, names=GT_COLUMNS)
    _df = _df.rename(
        columns={
            "frame": "FrameId",
            "id": "Id",
            "bb_left": "X",
            "bb_top": "Y",
            "bb_width": "Width",
            "bb_height": "Height",
            "conf": "Confidence",
        }
    )
    columns_to_int = ["FrameId", "Id", "X", "Y", "Width", "Height"]
    columns_to_float = ["Confidence"]
    _df[columns_to_int] = _df[columns_to_int].astype(int)
    _df[columns_to_float] = _df[columns_to_float].astype(float)
    return _df


def read_seqinfo(path: Union[str, pathlib.Path]) -> Dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    return dict(parser["Sequence"])


def combine_dataframes(
    dataframes: List[pd.DataFrame], n_frames: Optional[List[int]] = None
) -> pd.DataFrame:
    """Takes a list of single-camera dataframes and combines them for
    multi-camera evaluation.
    Args:
        dataframes (List[pd.DataFrame]): List of single-camera dataframes.
        n_frames (Optional[List[int]], optional): Defaults to None.
    Returns:
        pd.DataFrame: Multi-camera dataframe.
    """
    if n_frames is None:
        n_frames = [int(df["FrameId"].max()) for df in dataframes]
    count_frames = 0
    dfs = []
    for j, df in enumerate(dataframes):
        df["FrameId"] += count_frames
        count_frames += int(n_frames[j])
        dfs.append(df)
    return pd.concat(dfs).set_index(["FrameId", "Id"])


def evaluate(
    ground_truths: Dict[str, pd.DataFrame], predictions: Dict[str, pd.DataFrame]
):
    accs = []
    names = []
    for name, prediction in predictions.items():
        accs.append(
            mm.utils.compare_to_groundtruth(
                ground_truths[name], prediction, "iou", distth=0.5
            )
        )
        names.append(name)
    return accs, names


def evaluate_mtmc(
    data_paths: List[Union[str, pathlib.Path]],
    prediction_path: Union[str, pathlib.Path],
    scene_name: str,
):
    seqinfos = [read_seqinfo(os.path.join(path, "seqinfo.ini")) for path in data_paths]
    ground_truths = [
        read_txt(os.path.join(path, "gt", "gt.txt")) for path in data_paths
    ]
    prediction_paths = [
        os.path.join(prediction_path, seqinfo["name"] + ".txt") for seqinfo in seqinfos
    ]
    predictions = [read_txt(path) for path in prediction_paths]
    ground_truth_df = combine_dataframes(
        ground_truths, [seqinfo["seqlength"] for seqinfo in seqinfos]
    )
    prediction_df = combine_dataframes(
        predictions, [seqinfo["seqlength"] for seqinfo in seqinfos]
    )

    ground_truths = {scene_name: ground_truth_df}
    predictions = {scene_name: prediction_df}
    accs, names = evaluate(ground_truths, predictions)
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        names=names,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
    print(
        mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
    )


def evaluate_synthehicle_json(prediction, ground_truth):
    preds_to_eval = []
    truths_to_eval = []
    names = []
    for scene in ground_truth.keys():
        if scene in prediction.keys():
            gcams = ground_truth[scene]
            pcams = prediction[scene]
            preds_to_combine = []
            truths_to_combine = []
            for cam in gcams.keys():
                if cam in pcams.keys():
                    preds_to_combine.append(
                        mot_to_mm(
                            pd.DataFrame(prediction[scene][cam], columns=GT_COLUMNS)
                        )
                    )
                    truths_to_combine.append(
                        mot_to_mm(
                            pd.DataFrame(ground_truth[scene][cam], columns=GT_COLUMNS)
                        )
                    )
            names.append(scene)
            preds_to_eval.append(
                combine_dataframes(
                    preds_to_combine, n_frames=[1800] * len(preds_to_combine)
                )
            )
            truths_to_eval.append(
                combine_dataframes(
                    truths_to_combine, n_frames=[1800] * len(truths_to_combine)
                )
            )
    return evaluate_multi_scene(preds_to_eval, truths_to_eval, names)
