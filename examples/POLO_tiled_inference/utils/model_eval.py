import sys 
sys.path.append("/home/giacomo/projects/P0_YOLOcate/ultralytics/utils")

import json
import torch
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics.utils.metrics import DetMetrics, LocMetrics
from .globs import *


def compute_errors_img_lvl(gt_counts_dir: str, pred_counts_dir: str, class_ids: list, output_dir: str) -> dict:
    """
    Given a directory containing .json files of ground truth counts of an image set, and predicted counts, 
    compute the MAE and MSE. Also outputs an excel sheet containing the count differences per image.
    Arguments:
        gt_counts_dir (str):        path to the directory containing the gt counts.
        pred_counts_dir (str):      path to the directory containing the predicted counts.
        class_ids (list):           list of integer class ids. 
        output_dir (str):           path to a folder where the output can be stored.
    Returns:
        dictionary containing the MAE and MSE per class. 
    """

    # get paths to prediction files
    gt_files = [p for p in Path(gt_counts_dir).glob("*.json")]
    pred_files = [p for p in Path(pred_counts_dir).glob("*.json")]

    # make empty df to collect count differences
    df_cols = ["fn"]
    df_cols.extend([str(cls_id) for cls_id in class_ids])
    df_cols.append("binary")
    df_cols.extend([f"{str(cls_id)}_rel_diff" for cls_id in class_ids])
    df_cols.extend([f"{str(cls_id)}_rel_det" for cls_id in class_ids])
    df_cols.append("binary_rel_diff")
    df_cols.append("binary_rel_det")
    count_diffs_df = pd.DataFrame(columns=df_cols)

    # collect gt and predicted counts to plot later
    gts = {cls_id: [] for cls_id in class_ids}
    gts["binary"] = []
    preds = {cls_id: [] for cls_id in class_ids}
    preds["binary"] = []


    for fn_gt in gt_files:
        
        # get matching prediction file for a gt file 
        fn_pred = [pf for pf in pred_files if fn_gt.stem == pf.stem]

        assert len(fn_pred) <= 1, f"Found {len(fn_pred)} matching prediction files for file {fn_gt}. Expected 1 or 0."
        
        if fn_pred:
            fn_pred = fn_pred[0]
            with open(fn_pred, "r") as pred:
                pred_dict = json.load(pred)
        else:
            pred_dict = {str(cid): 0 for cid in class_ids}


        with open(fn_gt, "r") as gt:
            gt_dict = json.load(gt)

        # if no predictions for a class were made, the count number is zero 
        gt_dict = {int(k): v for k,v in gt_dict.items()}
        pred_dict = {int(k): v for k,v in pred_dict.items()}
        pred_dict_updated = {cls_id: 0 if cls_id not in pred_dict.keys() else pred_dict[cls_id] for cls_id in class_ids}
        
        # store predictions in list 
        for cls_id in class_ids:
            gts[cls_id].append(gt_dict[cls_id])
            gts["binary"].append(sum(gt_dict.values()))
            preds[cls_id].append(pred_dict_updated[cls_id])
            preds["binary"].append(sum(pred_dict_updated.values()))

        # get abs difference
        diffs = {str(cls_id): [gt_dict[cls_id] - pred_dict_updated[cls_id]] for cls_id in class_ids}
        diffs["binary"] = [sum(gt_dict.values()) - sum(pred_dict_updated.values())]
        diffs["fn"] = [fn_gt.stem]
        # get rel difference
        diffs_rel_diff = {f"{k}_rel_diff": [diffs[str(k)][0] / gt_dict[k]] if gt_dict[k] != 0 else diffs[str(k)][0] for k in gt_dict.keys()}
        diffs_rel_det = {f"{k}_rel_det": [pred_dict_updated[k] / gt_dict[k]] if gt_dict[k] != 0 else diffs[str(k)][0] for k in gt_dict.keys()}
        diffs_rel_diff["binary_rel_diff"] = [diffs["binary"][0] / sum(gt_dict.values())]
        diffs_rel_det["binary_rel_det"] = [sum(pred_dict_updated.values()) / sum(gt_dict.values())]

        # merge dicts 
        diffs.update(diffs_rel_diff)
        diffs.update(diffs_rel_det)
        # put into df
        row_df = pd.DataFrame(diffs)
        count_diffs_df = pd.concat([count_diffs_df, row_df], ignore_index=True)

    # get stats per class 
    summary_dict = {cls_id: {"MAE": count_diffs_df[f"{cls_id}"].abs().sum() / count_diffs_df.shape[0],
                             "ME":  count_diffs_df[f"{cls_id}"].sum() / count_diffs_df.shape[0],
                             "MSE": (count_diffs_df[f"{cls_id}"] ** 2).sum() / count_diffs_df.shape[0],
                             "MARE": count_diffs_df[f"{cls_id}_rel_diff"].abs().sum() / count_diffs_df.shape[0],
                             "MRE": count_diffs_df[f"{cls_id}_rel_det"].abs().sum() / count_diffs_df.shape[0]} for cls_id in class_ids}
    
    # get binary stats
    summary_dict["binary"] = {"MAE": count_diffs_df["binary"].abs().sum() / count_diffs_df.shape[0],
                               "ME":  count_diffs_df["binary"].sum() / count_diffs_df.shape[0],
                               "MSE": (count_diffs_df["binary"] ** 2).sum() / count_diffs_df.shape[0],
                               "MARE": count_diffs_df["binary_rel_diff"].abs().sum() / count_diffs_df.shape[0],
                               "MRE": count_diffs_df["binary_rel_det"].abs().sum() / count_diffs_df.shape[0]}
    
    # plot counts
    for k in gts.keys():
        plt.scatter(gts[k], preds[k], c="purple", s=5)
        #plt.yscale("log")
        #plt.xscale("log")
        plt.xlabel("True Count")
        plt.ylabel("Predicted Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/counts_gt_pred_{k}.png")
        plt.close()

    # output
    with open(f"{output_dir}/errors_img_lvl.json", "w") as f:
        json.dump(summary_dict, f, indent=1)

    count_diffs_df.to_excel(f"{output_dir}/count_diffs_img_lvl.xlsx")

    return summary_dict


def compute_em_img_lvl(preds_dir: str, class_id2name: dict, task: str, output_dir: str) -> dict:
    """
    Given a directory containing .npy files representing confusion matrices for an image set, compute
    a set of evaluation metrics. 
    Arguments:
        preds_dir (str):            path to the directory containing the .npy files.
        class_id2name (dict):       dict mapping class ids to class names.  
        task_id (str):              task string. Can either be "detect" or "locate"
        output_dir (str):           path to a folder where the output can be stored.
    Returns:
        dictionary containing evaluation metrics.
    """
    em_dict = {cat: {} for cat in class_id2name.keys()}

    # get paths to cm files
    cm_files = [p for p in Path(preds_dir).glob("*.npy")]
    cm_final = np.load(cm_files[0])
    cm_files = cm_files[1:]

    # aggregate single file cms
    for f in cm_files:
        cm_final += np.load(f)


     # get paths to pickel files
    stat_files = [p for p in Path(preds_dir).glob("*.pickle")]
    stats_glob = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    # initialize Metrics
    metrics = DetMetrics(names=class_id2name, save_dir=Path(output_dir), plot=True) if task == "detect" else \
              LocMetrics(names=class_id2name, save_dir=Path(output_dir), plot=True)

    # aggregate stat dicts
    for sf in stat_files:
        with open(sf, "rb") as f:
            stat_dict = pickle.load(f)

        if stat_dict.get("no_labels"):
            continue
        else:
            for k in stats_glob.keys():
                stats_glob[k].append(stat_dict[k])    

    # concatenate stats
    stats_final = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats_glob.items()}  # to numpy
    
    """
    check if predictions were made for an image. At this point, if there are no predictions, this means 
    that there must be false negatives, because cases where no predictions are made for an empty image 
    are accounted for a few lines above where I skip if stat+dict["no_labels"] is True. 
    """
    found_stuff = stats_final["tp"].any()
    if found_stuff:
        metrics.process(**stats_final)

    for cls_id in range(cm_final.shape[0] - 1):
        # confusion for each class = 1 - (TP / FP (without background))
        confusion = (1 - (cm_final[cls_id, cls_id] / (sum(cm_final[cls_id, :-1]) + 1e-6))) * 100
        precision = (cm_final[cls_id, cls_id] / (sum(cm_final[cls_id]) + 1e-6)) * 100
        recall = (cm_final[cls_id, cls_id] / (sum(cm_final[:, cls_id]) + 1e-6)) * 100

        em_dict[cls_id]["confusion"] = confusion
        em_dict[cls_id]["precision"] = precision
        em_dict[cls_id]["recall"] = recall
        em_dict[cls_id]["f1"] = ((2 * precision * recall) / (precision + recall + 1e-6))

        # collect YOLO/POLO metrics
        for i, k in enumerate(metrics.keys):
            if found_stuff:
                em_dict[cls_id][k] = metrics.class_result(cls_id)[i]     
            else:
                 em_dict[cls_id][k] = 0.0
                
    binary = {}
    precision_binary = (np.trace(cm_final) / (np.trace(cm_final) + sum(cm_final[:, -1]) + 1e-6)) * 100
    recall_binary = (np.trace(cm_final) / (np.trace(cm_final) + sum(cm_final[-1]) + 1e-6)) * 100

    binary["confusion"] = sum([em_dict[cls_id]["confusion"]for cls_id in em_dict.keys()]) / len(em_dict.keys())
    binary["precision"] = precision_binary
    binary["recall"] = recall_binary
    binary["f1"] = ((2 * precision_binary * recall_binary) / (precision_binary + recall_binary + 1e-6))
    # Add YOLO/POLO metrics
    for i, k in enumerate(metrics.keys):
        binary[k] = metrics.mean_results()[i]

    em_dict["binary"] = binary
    
    with open(f"{output_dir}/em.json", "w") as f:
        json.dump(em_dict, f, indent=1)

    return em_dict
