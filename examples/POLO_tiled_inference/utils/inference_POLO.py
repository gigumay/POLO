import shutil
import time
import tqdm
import json
import random
import requests
import math
import pickle

import torch
import cv2
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Union
from io import BytesIO
from collections import defaultdict
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils.ops import loc_nms, generate_radii_t
from ultralytics.utils.metrics import ConfusionMatrix, loc_dor_pw, box_iou

from .processing_utils import *
from .globs import *

PT_VIS_RADIUS = 3

N_RETRIES = 10
ERROR_NAMES_FOR_RETRY = ['ConnectionError']
RETRY_SLEEP_TIME = 0.01
IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}


def open_image(input_file: Union[str, BytesIO]) -> Image:
    """
    Opens an image in binary format using PIL.Image and converts to RGB mode. Taken from Dan Morris' 
    MegaDetector repo.
    
    Supports local files or URLs.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns:
        A PIL image object in RGB mode
    """
    
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        try:
            response = requests.get(input_file)
        except Exception as e:
            print(f'Error retrieving image {input_file}: {e}')
            success = False
            if e.__class__.__name__ in ERROR_NAMES_FOR_RETRY:
                for i_retry in range(0,N_RETRIES):
                    try:
                        time.sleep(RETRY_SLEEP_TIME)
                        response = requests.get(input_file)        
                    except Exception as e:
                        print(f'Error retrieving image {input_file} on retry {i_retry}: {e}')
                        continue
                    print('Succeeded on retry {}'.format(i_retry))
                    success = True
                    break
            if not success:
                raise
        try:
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise

    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L', 'I;16'):
        raise AttributeError(
            f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # Alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    #
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    #
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image



def load_img_gt(annotations: dict, boxes_in: bool, boxes_out: bool,  ann_format: str, device=torch.device, box_dims: dict = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expects the annotations in an image as a dictionary of dictionaries, and from that creates tensors for the ground truth labels. 
    Arguments:
        annotations (dictionary):           dictionary containing all annotations (as dictionaries) within the image under consideration.
        boxes_in (bool):                    if true, ground truth labels are expected as box annotations
        boxes_out (bool):                   if true, retrieved labels will be returned as boxes
        ann_format (string):                string indicatin the annotation format
        task (string):                      string indicating the prediction task
        device (torch.device):              device on which to store the gt tensors
        box_dims (dict):                    dimensions of ground-truth bounding boxes
    Returns:
        tuple[torch.Tensor, torch.Tensor]:  the ground truth coordinates and classes
    """
    coords_list = []
    cls_list = []
    n_coords = 4 if boxes_out else 2
    for ann in annotations:
        label = ann[DATA_ANN_FORMATS[ann_format]["label_key"]]
        cat_id = ann[DATA_ANN_FORMATS[ann_format]["category_key"]]

        if boxes_out:
            x_center = label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["width_idx"]] / 2.0)
            y_center = label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["height_idx"]] / 2.0)
            
            box_dims_checked = {}
            if box_dims:
                box_dims_checked["width"] = box_dims[cat_id]["width"]
                box_dims_checked["height"] = box_dims[cat_id]["height"]
            else: 
                box_dims_checked["width"] = label[DATA_ANN_FORMATS[ann_format]["width_idx"]]
                box_dims_checked["height"] = label[DATA_ANN_FORMATS[ann_format]["height_idx"]]
            
            xmin = x_center - (box_dims_checked["width"]/2.0)
            xmax = x_center + (box_dims_checked["width"]/2.0)
            ymin = y_center - (box_dims_checked["height"]/2.0)
            ymax = y_center + (box_dims_checked["height"]/2.0)
            coords_list.append([xmin, ymin, xmax, ymax])
        else:
            if boxes_in:
                x = label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["width_idx"]] / 2.0)
                y = label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["height_idx"]] / 2.0) 
            else: 
                x = label[DATA_ANN_FORMATS[ann_format]["x_idx"]]
                y = label[DATA_ANN_FORMATS[ann_format]["y_idx"]]

            coords_list.append([x,y])
    
        cls_list.append(ann["category_id"])

    if not coords_list:
        coords_t = torch.empty((0, n_coords), dtype=torch.float)
        cls_t = torch.empty((0, 1))
    else: 
        coords_t = torch.tensor(coords_list, device=device, dtype=torch.float)
        cls_t = torch.tensor(cls_list, device=device)

    return coords_t, cls_t 


def collect_boxes(predictions: list, class_ids: list, patches: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all patch-level boxes from a list of predictions and map them to the image space. If an output directory 
    is specified, patch level counts will be writtento a .json file in that directory. 
    Arguments:
        predictions (list):         list containing all patch predictions.
        class_ids (list):           list of class ids. 
        patches (list):             list containing all patches.
        device (torch.device):      device used for inference.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the coordinates, confidence scores and classes of the image level predictions

    """

    all_preds = torch.empty((0, 6), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.boxes

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordinates, confidence and class into one tensor
        data_merged = torch.hstack((data.xyxy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # map prediction to image_level
        data_merged[:, [0, 2]] = data_merged[:, [0,2]] + patches[i]["coords"]["x_min"]
        data_merged[:, [1, 3]] = data_merged[:, [1,3]] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((4, 1, 1), 1)


def collect_locations(predictions: list, class_ids: list, patches: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all patch-level locations from a list of predictions and map them to the image space. If an output directory 
    is specified, patch level counts will be writtento a .json file in that directory. 
    Arguments:
        predictions (list):         list containing all patch predictions.
        class_ids (list):           list of class ids. 
        patches (list):             list containing all patches.
        device (torch.device):      device used for inference.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the coordinates, confidence scores and classes of the image level predictions

    """

    all_preds = torch.empty((0, 4), device=device)

    for i, patch_preds in enumerate(predictions):
        data = patch_preds.locations

        assert patches[i]["patch_fn"] == patch_preds.path

        # merge bbox coordiantes, confidence and class into one tensor
        data_merged = torch.hstack((data.xy, data.conf.unsqueeze(1), data.cls.unsqueeze(1)))

        # map prediction to image_level
        data_merged[:, 0] = data_merged[:, 0] + patches[i]["coords"]["x_min"]
        data_merged[:,1] = data_merged[:, 1] + patches[i]["coords"]["y_min"]

        # combine all patch predictions into one image level tensor
        all_preds = torch.vstack((all_preds, data_merged))
    
    return all_preds.split((2, 1, 1), 1)


# copied from the ultralytics repo (validator.py) to avoid circular inputs
def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        iouv = torch.linspace(0.5, 0.95, 10)
        correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True

        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    

# copied from the ultralytics repo (validator.py) to avoid circular inputs
def match_predictions_loc(pred_classes, true_classes, dor, use_scipy=False):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using DoR.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        dor (torch.Tensor): An NxM tensor containing the pairwise DoR values for predictions and ground truth
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 DoR thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - DoR thresholds
    dorv = torch.linspace(1.0, 0.1, 10)
    correct = np.zeros((pred_classes.shape[0], dorv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    dor = dor * correct_class  # zero out the wrong classes
    dor = dor.cpu().numpy()
    for i, threshold in enumerate(dorv.cpu().tolist()):
        if use_scipy:
            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
            import scipy  # scope import to avoid importing for all commands

            cost_matrix = dor * (dor <= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero((dor <= threshold) & (dor > 0))  # DoR < threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[dor[matches[:, 0], matches[:, 1]].argsort()]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    

def plot_img_predictions(img_fn: str, coords: torch.Tensor, cls: torch.Tensor, output_dir: str, gt_coords: torch.Tensor = None, gt_class: torch.Tensor = None,
                         plot_gt_color: bool = False, pre_nms: bool = False) -> None:
    """
    Plot an image and visualize the corresponding predictions. If provided, the ground truth is plotted as well. 
    Arguments:
        img_fn (str):               path to the imnage file
        coords (torch.Tensor):      tensor containing the prediction coordinates. Either 4 (boxes in xyxy format) or two (points) columns. 
        cls (torch.Tensor):         tensor containing the predicted classes. 
        output_dir (str):           path to the folder where the plot can be stored.
        gt_coords (torch.Tensor):   tensor containing the ground truth coordinates.
        gt_class (torch.Tensor):    tensor containing the ground truth classes.
        plot_gt_color (bool):       if True, ground truth annotations will be plotted in colors of the correpsonding class.
        pre_nms (bool):             if True, the coords tensor is assumed to contain predictions before global NMS was applied.

    """

    img_arr = cv2.imread(img_fn)
    boxes = coords.shape[1] == 3

    for i in range(coords.shape[0]):
        if boxes:
            cv2.rectangle(img=img_arr, pt1=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                          pt2=(int(round(coords[i, 2].item())), int(round(coords[i, 3].item()))), color=CLASS_COLORS[int(cls[i].item())], 
                          thickness=3)
        else:
            cv2.circle(img=img_arr, center=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                       radius=PT_VIS_RADIUS, color=CLASS_COLORS[int(cls[i].item())], thickness=-1)
            
    if gt_coords is not None:
        for i in range(gt_coords.shape[0]):
            gt_color = (0, 255, 0) if not plot_gt_color else CLASS_COLORS[int(gt_class[i].item())]
            if boxes:
                cv2.rectangle(img=img_arr, pt1=(int(round(gt_coords[i, 0].item())), int(round(gt_coords[i, 1].item()))),
                              pt2=(int(round(gt_coords[i, 2].item())), int(round(gt_coords[i, 3].item()))), color=gt_color, thickness=3)
            else:
                cv2.circle(img=img_arr, center=(int(round(gt_coords[i, 0].item())), int(round(gt_coords[i, 1].item()))),
                           radius=PT_VIS_RADIUS, color=gt_color, thickness=-1)

    output_ext = "_pre_nms" if pre_nms else ""
            
    assert cv2.imwrite(f"{output_dir}/{Path(img_fn).stem}{output_ext}.jpg", img_arr), "Plotting failed!"


def run_tiled_inference_POLO(model: tuple[str, YOLO], class_ids: list, radii: dict, dor_thresh: float, imgs_dir: str, img_files_ext: str, 
                             patch_dims: dict, patch_overlap: float, output_dir: str, ann_file: str = None, ann_format: str = None, 
                             box_dims: dict = None, device: int = "cuda:0", vis_prob: float = -1.0, vis_density: int = math.inf, 
                             plot_gt: bool = False, plot_gt_color: bool = False, patch_quality: int = 95, save_pre_output: bool = False, 
                             rm_tiles: bool = True, verbose: bool = False) -> None:
    """
    Perform tiled inference on a directory of images. If ground truth annotations are available, produces confusion matrices and other output 
    to compute evaluation matrices. 
    Arguments:
        model (tuple[str, YOLO]):   path to the model file (.pt) to be used for inference, or already constructed model object. 
        class_ids (list):           list of integer class IDs in the data the model was trained on.
        radii (dict):               dictionary containing the radius values for localization NMS. 
        dor_thresh (float):         DoR threshold to use for NMS during inference and when stitching tiles together. 
        imgs_dir (str):             path to the folder containing the images inference needs to be performed on.
        img_files_ext (str):        Extension of image files. E.g., 'JPG'.
        patch_dims (dict):          dictionary with keys 'width' & 'height' specifying the tile dimensions.
        patch_overlap (float):      amount of overlap (fraction) between tiles. 
        output_dir (str):           path to the directory the output will be stored in.
        ann_file (string):          path to the file containing the annotations for the images to be processed. The file is
                                    expected to contain a dictionary that maps image filenames ot annotations, such as
                                    produced by the methods in preprocessing.py (DC11 Utils)
        ann_format (string):        format of the annotations.
        box_dims (dict):            dimensions of the bounding boxes used for model training. For cases where 
                                    dimensions that differ from what is specified in the annotations were used. 
        device (str):               device onto which to load the model. Defaults to "cuda:0"
        vis_prob (str):             plotting probability. For each image, a random number between 0 and 1 is drawn, 
                                    and if it's below vis_prob, the image in question & the corresponding predictions
                                    will be visualized. 
        vis_density (int):          number of objects/animals that will trigger plotting. If the number of predictions 
                                    for an image is equal to or exceeds vis_prob, the image & predictions will be plotted
                                    regardless of vis_prob.
        plot_gt (bool):             if True, ground truth annotations will be plotted as well.
        plot_gt_color (bool):       if True, ground truth annotations will be plotted in colors of the correpsonding class.
        patch_quality (float):      quality of tle image files (not sure what this does or why I need it tbh).
        save_pre_output (bool):     if True, outputs (plots + detections) before NMS will be sav ed as well.
        rm_tiles (bool):            whether to remove the tile image files after inference.
        verbose (bool):             if True, the amount of predictions removed by NMS will be printed to the console.
    Returns:
        None
    """

    assert patch_overlap < 1 and patch_overlap >= 0, \
        'Illegal tile overlap value {}'.format(patch_overlap)
     
    # make directory for storing tiles
    tiling_dir = f"{imgs_dir}/tiles"
    Path(tiling_dir).mkdir()

    # make output folders
    det_dir = f"{output_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}"
    Path(det_dir).mkdir(exist_ok=True)

    if vis_prob > 0 or vis_density < math.inf: 
        vis_dir = f"{output_dir}/vis"
        Path(vis_dir).mkdir(exist_ok=True)
 
    mdl = YOLO(model).to(device) if isinstance(model, str) else model.to(device)
   
    # read annotations file if available
    if ann_file is not None:
        with open(ann_file, "r") as f:
            ann_dict = json.load(f)
    else:
        ann_dict = None

    img_fns = list(Path(imgs_dir).glob(f"*.{img_files_ext}"))
    # for accumulating total counts across images
    counts_sum = {cls_id: 0 for cls_id in class_ids}    
    
    print("*** Processing images")
    for fn in tqdm(img_fns, total=len(img_fns)):
        im = open_image(fn)
        patch_start_positions = get_patch_start_positions(img_width=im.width, img_height=im.height, patch_dims=patch_dims, 
                                                          overlap=patch_overlap) 
        
        patches = []
        # create tiles to perform inference on
        for patch in patch_start_positions: 
            patch_coords = {"x_min": patch[0], 
                            "y_min": patch[1], 
                            "x_max": patch[0] + patch_dims["width"] - 1,
                            "y_max": patch[1] + patch_dims["height"] - 1}
            
            patch_name = patch_info2name(image_name=fn.stem, patch_x_min=patch_coords['x_min'], patch_y_min=patch_coords['y_min'])
            patch_fn = f"{tiling_dir}/{patch_name}.jpg"
            
            patch_metadata = {"patch_fn": patch_fn,
                              "patch_name": patch_name,
                              "coords": patch_coords}
            
            patches.append(patch_metadata)
        
            patch_im = im.crop((patch_coords["x_min"], patch_coords["y_min"], patch_coords["x_max"] + 1,
                                patch_coords["y_max"] + 1))
            assert patch_im.size[0] == patch_dims["width"]
            assert patch_im.size[1] == patch_dims["height"]
        
            assert not Path(patch_fn).exists()
            patch_im.save(patch_fn, quality=patch_quality)


        # run detection on patches 
        patch_fns = [patch["patch_fn"] for patch in patches]
        predictions = mdl(patch_fns, radii=radii, dor=dor_thresh, verbose=False)
    
        # collect predictions from each patch and map it back to image level
        coords, conf, cls = collect_locations(predictions=predictions, class_ids=class_ids, patches=patches, device=mdl.device)
        
        # get counts before nms
        pre_nms = coords.shape[0]

        cls_idx_pre, counts_pre = torch.unique(cls.squeeze(1), return_counts=True)
        counts_pre_dict = {}
        for j in range(cls_idx_pre.shape[0]):
            counts_pre_dict[int(cls_idx_pre[j].item())] = int(counts_pre[j].item())

        if save_pre_output:
            with open(f"{det_dir}/{fn.stem}_pre_nms.json", "w") as f:
                json.dump(counts_pre_dict, f, indent=1)

        # perform nms
        radii_preds_t = generate_radii_t(radii=radii, cls=cls.squeeze(1))
        idxs = loc_nms(preds=coords, scores=conf.squeeze(1), radii=radii_preds_t, dor_thres=dor_thresh)
        
        # combine coordinates, confidence and class into one tensor
        preds_img_final = torch.hstack((coords[idxs], conf[idxs], cls[idxs]))

        
        # If annotations are available, collect evaluation metrics at the image level 
        if ann_file is not None:
            assert ann_format is not None, "Please provide the format of the annotations file you passed with the 'ann_file' parameter."

            boxes_in = "BX" in ann_format
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[fn.stem], boxes_in=boxes_in, boxes_out=False, ann_format=ann_format, 
                                            device=coords.device, box_dims=box_dims)
            radii_gt_t = generate_radii_t(radii=radii, cls=gt_cls)


            # Make Confusion matrix
            cfm_img = ConfusionMatrix(nc=len(class_ids), task="locate", dor_thresh=dor_thresh)
            cfm_img.process_batch_loc(localizations=preds_img_final, gt_locs=gt_coords, gt_cls=gt_cls, radii=radii_gt_t)
            # write confusion matrix to file
            with open(f"{det_dir}/{fn.stem}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)


            # make stats dict
            npr = preds_img_final.size(dim=0)
            stat = dict(
                conf=torch.zeros(0, device=preds_img_final.device),
                pred_cls=torch.zeros(0, device=preds_img_final.device),
                tp=torch.zeros(npr, 10, dtype=torch.bool, device=preds_img_final.device),
                no_labels=False
            )
            nl = gt_cls.size(dim=0)
            stat["target_cls"] = gt_cls

            if npr != 0: 
                stat["conf"] = preds_img_final[:, 2]
                stat["pred_cls"] = preds_img_final[:, 3]
                # Evaluate
                if nl:
                    dor = loc_dor_pw(loc1=gt_coords, loc2=preds_img_final[:, :2], radii=radii_gt_t)
                    stat["tp"] = match_predictions_loc(pred_classes=preds_img_final[:, 3], true_classes=gt_cls, dor=dor)    
            else:
                if nl == 0: 
                    stat["no_labels"] = True

            #write to pickle file:
            with open(f"{det_dir}/{fn.stem}_stats.pickle", "wb") as f:
                pickle.dump(stat, f)
        else:
            gt_coords = None 
            gt_cls = None
            radii_gt_t = None
        
        # get counts post nms 
        post_nms = preds_img_final.shape[0]
        cls_idx_post, counts_post = torch.unique(preds_img_final[:, -1], return_counts=True)
        counts_post_dict = {}
        for j in range(cls_idx_post.shape[0]):
            counts_post_dict[int(cls_idx_post[j].item())] = int(counts_post[j].item())

        with open(f"{det_dir}/{fn.stem}.json", "w") as f:
            json.dump(counts_post_dict, f, indent=1)
        
        if verbose:
            print(f"NMS removed {pre_nms - post_nms} predictions!")

        # add to count sum
        for class_idx, n in counts_post_dict.items():
            counts_sum[class_idx] += n

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (post_nms >= vis_density)

        if visualize:
            assert not (plot_gt and ann_file is None), "Plotting ground truth requested but no annotations file provided!"
            plot_img_predictions(img_fn=str(fn), coords=coords[idxs], cls=cls[idxs], pre_nms=False, output_dir=vis_dir, 
                                 gt_coords=gt_coords, gt_class=gt_cls, plot_gt_color=plot_gt_color)
            if save_pre_output:
                plot_img_predictions(img_fn=str(fn), coords=coords, cls=cls, pre_nms=True, output_dir=vis_dir, 
                                    gt_coords=gt_coords, gt_class=gt_cls, plot_gt_color=plot_gt_color)

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)

    if rm_tiles:
        shutil.rmtree(tiling_dir) 



def read_output_SAHI(out_json_SAHI: str, dataset_json_SAHI: str, class_ids: list, iou_thresh: float, output_dir: str,  ann_file: str = None,
                     ann_format: str = None, box_dims: dict = None, vis_prob: float = -1.0, vis_density: int = math.inf, plot_gt: bool = False, 
                     plot_gt_color: bool = False) -> None:
    """
    Given a SAHI output file, make a detections directory that for each image contains a .json file with counts and (if 
    also given gorund truth annotations) confusion matrices in the shape of a .npy file. 
    Arguments:
        out_json_SAHI (str):            path to the SAHI output file.
        dataset_json_SAHI (str):        path to the SAHI dataset file.
        class_ids (list):               list of class ids.
        iou_thresh (float):             iou threshold to apply during postprocessing.
        output_dir (str):               path to the output directory.
        ann_file (str):                 path to the file containing the annotations for the images to be processed. The file is
                                        expected to contain a dictionary that maps image filenames ot annotations, such as
                                        produced by the methods in preprocessing.py (DC11 Utils)
        ann_format (string):            format of the annotations.
        box_dims (dict):                dimensions of the bounding boxes used for model training. For cases where 
                                        dimensions that differ from what is specified in the annotations were used. 
        device (str):                   device onto which to load the model. Defaults to "cuda:0"
        vis_prob (str):                 plotting probability. For each image, a random number between 0 and 1 is drawn, 
                                        and if it's below vis_prob, the image in question & the corresponding predictions
                                        will be visualized. 
        vis_density (int):              number of objects/animals that will trigger plotting. If the number of predictions 
                                        for an image is equal to or exceeds vis_prob, the image & predictions will be plotted
                                        regardless of vis_prob.
        plot_gt (bool):                 if True, ground truth annotations will be plotted as well.
        plot_gt_color (bool):           if True, ground truth annotations will be plotted in colors corresponding to the predicted class.
    """
    dets_dir = f"{output_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}"
    Path(dets_dir).mkdir(exist_ok=True)

    if vis_prob > 0 or vis_density < math.inf: 
        vis_dir = f"{output_dir}/vis"
        Path(vis_dir).mkdir(exist_ok=True)
    
    with open(out_json_SAHI, "r") as out, open(dataset_json_SAHI, "r") as data:
        out_SAHI = json.load(out)
        dataset_SAHI = json.load(data)

    if ann_file is not None:
        with open(ann_file, "r") as anns:
            ann_dict = json.load(anns)

    id2preds = {}
    counts_total = {cid: 0 for cid in class_ids}

    # for each image in the dataset
    for img in dataset_SAHI["images"]:
        detections = []
        counts = defaultdict(int)
        img_id = img["id"]
        img_path = img["file_name"]
        img_fn = Path(img_path).stem

        # extract and group predictions 
        for det in out_SAHI:
            if det["image_id"] == img_id:
                xmin = det["bbox"][0]
                ymin = det["bbox"][1]
                xmax = xmin + det["bbox"][2]
                ymax = ymin + det["bbox"][3]

                conf = det["score"]
                cat = det["category_id"]
                counts[int(cat)] += 1

                detections.append([xmin, ymin, xmax, ymax, conf, cat])

        id2preds[img_id] = detections
        dets_t = torch.tensor(detections, dtype=torch.float, device=torch.device("cuda")) if detections else None

        with open(f"{dets_dir}/{img_fn}.json", "w") as f:
            json.dump(counts, f, indent=1)

        for cid in counts.keys():
            counts_total[cid] += counts[cid]

        # if annotations are available, make CMs and extract EM
        if ann_file is not None:
            cfm_img = ConfusionMatrix(nc=len(class_ids), task="detect", iou_thres=iou_thresh)
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[img_fn], boxes_in=True, boxes_out=True, ann_format=ann_format, 
                                            device=torch.device("cuda"), box_dims=box_dims)
            
            cfm_img.process_batch(detections=dets_t, gt_bboxes=gt_coords, gt_cls=gt_cls)

            # write confusion matrix to file
            with open(f"{dets_dir}/{img_fn}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)


            # make stats dict
            npr = dets_t.size(dim=0) if dets_t is not None else 0
            stat = dict(
                conf=torch.zeros(0, device=gt_coords.device),
                pred_cls=torch.zeros(0, device=gt_coords.device),
                tp=torch.zeros(npr, 10, dtype=torch.bool, device=gt_coords.device),
            )
            
            nl = gt_cls.size(dim=0)
            stat["target_cls"] = gt_cls

            if npr != 0: 
                stat["conf"] = dets_t[:, 4]
                stat["pred_cls"] = dets_t[:, 5]
                # Evaluate
                if nl:
                    iou = box_iou(gt_coords, dets_t[:, :4])
                    stat["tp"] =  match_predictions(dets_t[:, 5], gt_cls, iou)        
            else:
                if not nl: 
                    raise ValueError("No predictions and no labels case is skipped in the original code, but I need a matching file, so I'm adding zero-stats.\n" \
                                        "This is going to wrongly reduce the performance metrics. Normally this shouldn't happen as there shouldn't be empty images.\n" \
                                        "This error being raised, however, means that there are!!")

            #write to pickle file:
            with open(f"{dets_dir}/{img_fn}_stats.pickle", "wb") as f:
                pickle.dump(stat, f)
        else:
            gt_coords = None
            gt_cls = None
                

        post_nms = 0 if dets_t is None else dets_t.shape[0]
        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (post_nms >= vis_density)

        if visualize and post_nms > 0:
            assert not (plot_gt and ann_file is None), "Plotting ground truth requested but no annotation file provided!"
            plot_img_predictions(img_fn=img_path, coords=dets_t[:,:4], cls=dets_t[:,-1], output_dir=vis_dir, 
                                 gt_coords=gt_coords, gt_class=gt_cls, plot_gt_color=plot_gt_color)  

    with open(f"{output_dir}/counts_toal.json", "w") as f:
        json.dump(counts_total, f)
      
            


def img_inference(model: str | YOLO, class_ids: list, imgs_dir: str, img_files_ext: str, output_dir: str, ann_file: str = None, 
                  ann_format: str = None, dor_thresh: float = None, iou_thresh: float = None, radii: dict = None, box_dims: dict = None, 
                  device: str = "cuda:0", vis_prob: float = -1.0, vis_density: int = math.inf, plot_gt: bool = False, plot_gt_color: bool = False) -> None:
    """
    Perform inference on a directory of images. If ground truth annotations are available, produces confusion matrices and other output 
    to compute evaluation matrices. 
    Arguments:
        model (str | YOLO):         path to the model file (.pt) to be used for inference, or already constructed model object. 
        class_ids (list):           list of integer class IDs in the data the model was trained on.
        imgs_dir (str):             path to the folder containing the images inference needs to be performed on.
        img_files_ext (str):        Extension of image files. E.g., 'JPG'.
        output_dir (str):           path to the directory the output will be stored in.
        ann_file (string):          path to the file containing the annotations for the images to be processed. The file is
                                    expected to contain a dictionary that maps image filenames ot annotations, such as
                                    produced by the methods in preprocessing.py (DC11 Utils)
        ann_format (string):        format of the annotations.
        dor_thresh (float):         DoR threshold to use for NMS during inference.
        iou_thresh (float):         IoU threshold to use for NMS during inference.
        radii (dict):               dictionary containing the radius values for localization NMS. 
        box_dims (dict):            dimensions of the bounding boxes used for model training. For cases where 
                                    dimensions that differ from what is specified in the annotations were used. 
        vis_prob (str):             plotting probability. For each image, a random number between 0 and 1 is drawn, 
                                    and if it's below vis_prob, the image in question & the corresponding predictions
                                    will be visualized. 
        vis_density (int):          number of objects/animals that will trigger plotting. If the number of predictions 
                                    for an image is equal to or exceeds vis_prob, the image & predictions will be plotted
                                    regardless of vis_prob.
        plot_gt (bool):             if True, ground truth annotations will be plotted as well.
        plot_gt_color (bool):       if True, ground truth annotations will be plotted in colors corresponding to the predicted class.
    Returns:
        None
    """

    # make output folders
    det_dir = f"{output_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}"
    Path(det_dir).mkdir(exist_ok=True)

    if vis_prob > 0 or vis_density < math.inf: 
        vis_dir = f"{output_dir}/vis"
        Path(vis_dir).mkdir(exist_ok=True)
 
    mdl = YOLO(model).to(device) if isinstance(model, str) else model.to(device)
    boxes = mdl.task == "detect"
    conf_idx = 4 if boxes else 2
    cls_idx = 5 if boxes else 3
    coords_lim = 4 if boxes else 2
   
    # read annotations file if available 
    if ann_file is not None:
        with open(ann_file, "r") as f:
            ann_dict = json.load(f)

    img_fns = list(Path(imgs_dir).glob(f"*.{img_files_ext}"))
    
    # for accumulating total counts across images
    counts_sum = {cls_id: 0 for cls_id in class_ids}    

    # run detection on images
    print("*** Performing inference..")
    if boxes:
        predictions = mdl(img_fns, iou=iou_thresh, verbose=False)
    else:
        predictions = mdl(img_fns, radii=radii, dor=dor_thresh, verbose=False)

    print("*** Collecting results...")
    for i, img_preds in enumerate(tqdm(predictions)):
        img_p = img_preds.path
        data = img_preds.boxes if boxes else img_preds.locations

        coords = data.xyxy if boxes else data.xy
        conf = data.conf.unsqueeze(1)
        cls = data.cls.unsqueeze(1)

        # combine coordinates, confidence and class into one tensor
        preds_t = torch.hstack((coords, conf, cls))

        # If annotations are available, collect evaluation metrics at the image level 
        if ann_file is not None:
            boxes_in = "BX" in ann_format
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[Path(img_p).stem], boxes_in=boxes_in, boxes_out=boxes, ann_format=ann_format, 
                                            device=coords.device, box_dims=box_dims)
            
            # Make Confusion matrix
            if boxes:
                radii_gt_t = None
                cfm_img = ConfusionMatrix(nc=len(class_ids), iou_thresh=iou_thresh)
                cfm_img.process_batch(detectiosn=preds_t, gt_locs=gt_coords, gt_cls=gt_cls)
            else:
                radii_gt_t = generate_radii_t(radii=radii, cls=gt_cls)
                cfm_img = ConfusionMatrix(nc=len(class_ids), task="locate", dor_thresh=dor_thresh)
                cfm_img.process_batch_loc(localizations=preds_t, gt_locs=gt_coords, gt_cls=gt_cls, radii=radii_gt_t)

            # write confusion matrix to file
            with open(f"{det_dir}/{Path(img_p).stem}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)


            npr = preds_t.size(dim=0)
            stat = dict(
                conf=torch.zeros(0, device=preds_t.device),
                pred_cls=torch.zeros(0, device=preds_t.device),
                tp=torch.zeros(npr, 10, dtype=torch.bool, device=preds_t.device),
                no_labels=False
            )
            nl = gt_cls.size(dim=0)
            stat["target_cls"] = gt_cls

            if npr != 0: 
                stat["conf"] = preds_t[:, conf_idx]
                stat["pred_cls"] = preds_t[:, cls_idx]
                # Evaluate
                if nl:
                    if boxes:
                        iou = box_iou(box1=gt_coords, box2=preds_t[:, :coords_lim])
                        stat["tp"] = match_predictions(pred_classes=preds_t[:, cls_idx], true_classes=gt_cls, iou=iou)
                    else:
                        dor = loc_dor_pw(loc1=gt_coords, loc2=preds_t[:, :coords_lim], radii=radii_gt_t)
                        stat["tp"] = match_predictions_loc(pred_classes=preds_t[:, cls_idx], true_classes=gt_cls, dor=dor)    
            else:
                if nl == 0: 
                    stat["no_labels"] = True


            #write to pickle file:
            with open(f"{det_dir}/{Path(img_p).stem}_stats.pickle", "wb") as f:
                pickle.dump(stat, f)
        else:
            gt_coords = None
            gt_cls = None
        
        # get counts post nms 
        post_nms = preds_t.shape[0]
        cls_idx_post, counts_post = torch.unique(preds_t[:, -1], return_counts=True)
        counts_post_dict = {}
        for j in range(cls_idx_post.shape[0]):
            counts_post_dict[int(cls_idx_post[j].item())] = int(counts_post[j].item())

        with open(f"{det_dir}/{Path(img_p).stem}.json", "w") as f:
            json.dump(counts_post_dict, f, indent=1)
        
        # add to count sum
        for class_idx, n in counts_post_dict.items():
            counts_sum[class_idx] += n

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (post_nms >= vis_density)

        if visualize:
            assert not (plot_gt and ann_file is None), "Plotting ground truth requested but no annotation file provided!"
            plot_img_predictions(img_fn=img_p, coords=coords, cls=cls, pre_nms=False, output_dir=vis_dir, 
                                 gt_coords=gt_coords, gt_class=gt_cls, plot_gt_color=plot_gt_color)

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)
