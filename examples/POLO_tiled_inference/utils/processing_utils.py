import cv2
import math
import json

from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from .globs import *


IMG_SUFFIXES = [".PNG", ".JPG", ".JPEG", ".jpg"]

def get_patch_start_positions(img_width: int, img_height: int, patch_dims: dict, overlap: float) -> list: 
    """
    Creates list of the starting positions (in pixel offsets) of each patch in a given image. 
    Arguments:
        img_width (int):    width of the image in pixels
        img_height (int):   height of the image in pixels
        patch_dims (dict):  Dict containing the dimensions of the patches.
    Returns:
        list of starting positions
    """
   
    patch_stride = {"x": round(patch_dims["width"]*(1.0-overlap)), "y": round(patch_dims["height"]*(1.0-overlap))}

    patch_width = patch_dims["width"]
    patch_height = patch_dims["height"]

    if patch_width >= img_width:
        print(f"WARNING: patch width ({patch_width}) is bigger than image width ({img_width}).\n" \
              f"Setting image width as pacth width!")
        patch_width = img_width
    
    if patch_height >= img_height:
        print(f"WARNING: patch width ({patch_height}) is bigger than image width ({img_height}).\n" \
              f"Setting image width as pacth width!")
        patch_height = img_height

    
    def add_patch_row(patch_start_positions,y_start):
        """
        Add one row to the list of patch start positions, i.e. loop over all columns.
        """
        x_start = 0; x_end = x_start + patch_width - 1
        
        while(True):
            patch_start_positions.append([x_start,y_start])
            
            if x_end == img_width - 1:
                break
            
            # Move one patch to the right
            x_start += patch_stride["x"]
            x_end = x_start + patch_width - 1
             
            # If this patch flows over the edge, add one more patch to cover the pixels at the end
            if x_end > (img_width - 1):
                overshoot = (x_end - img_width) + 1
                x_start -= overshoot
                x_end = x_start + patch_width - 1
                patch_start_positions.append([x_start,y_start])
                break
            
        return patch_start_positions
        
    patch_start_positions = []
    
    y_start = 0; y_end = y_start + patch_height - 1
    
    while(True):
        patch_start_positions = add_patch_row(patch_start_positions,y_start)
        
        if y_end == img_height - 1:
            break
        
        # Move one patch down
        y_start += patch_stride["y"]
        y_end = y_start + patch_height - 1
        
        # If this patch flows over the bottom, add one more patch to cover the pixels at the bottom
        if y_end > (img_height - 1):
            overshoot = (y_end - img_height) + 1
            y_start -= overshoot
            y_end = y_start + patch_height - 1
            patch_start_positions = add_patch_row(patch_start_positions,y_start)
            break
    
    for p in patch_start_positions:
        assert p[0] >= 0 and p[1] >= 0 and p[0] <= img_width and p[1] <= img_height, \
        f"Patch generation error (illegal patch {p})!"
        
    # The last patch should always end at the bottom-right of the image
    assert patch_start_positions[-1][0] + patch_width == img_width, \
        "Patch generation error (last patch does not end on the right)"
    assert patch_start_positions[-1][1] + patch_height == img_height, \
        "Patch generation error (last patch does not end at the bottom)"
    
    # All patches should be unique
    patch_start_positions_tuples = [tuple(x) for x in patch_start_positions]
    assert len(patch_start_positions_tuples) == len(set(patch_start_positions_tuples)), \
        "Patch generation error (duplicate start position)"
    
    return patch_start_positions

    

def patch_info2name(image_name: str, patch_x_min: int, patch_y_min: int, is_empty: bool = None) -> str:
    """
    Generates the name of a patch.
    Arguments: 
        image_name (str):   name of the image that contains the patch
        patch_x_min (int):  x coordinate of the left upper corner of the patch within the 
                            original image
        patch_y_min (int):  y coordinate of the left upper corner of the patch within the 
                            original image 
        is_empty (bool):    indicates whether the patch is empty
    Returns: 
        string containing the patch name
    """
    empty_ext = "_empty" if is_empty else ""
    patch_name = f"{image_name}_{str(patch_x_min).zfill(4)}_{str(patch_y_min).zfill(4)}{empty_ext}"
    return patch_name



def vis_img(img_path: str, annotations: list, ann_format: str, boxes_in: bool, boxes_out: bool, output_dir: str) -> None:
    """
    Plot an image with its labels.
    Arguments:
        img_path (str):             path to the image to be plotted
        annotations (list):         list of labels wihtin the image
        ann_format (str):           format in which the annotations are provided
        boxes_in (bool):            if true, annotations are expected to be box labels
        boxes_out (bool):           if true, annotations are plotted as boxes. Requires boxes_in to be true. 
    Returns:
        None
    """
    img_arr = cv2.imread(img_path)
    img_name = Path(img_path).stem

    for ann in annotations:
        label = ann[DATA_ANN_FORMATS[ann_format]["label_key"]]
        cls_id = ann[DATA_ANN_FORMATS[ann_format]["category_key"]]
        
        if boxes_out:
            assert boxes_in, "Cannot plot boxes from point labels"
            #cv2 needs int coordinates
            xmin_img = round(label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]])
            xmax_img = round(label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["width_idx"]])
            ymin_img = round(label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]])
            ymax_img = round(label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["height_idx"]])
         
            cv2.rectangle(img=img_arr, pt1=(xmin_img, ymin_img), pt2=(xmax_img, ymax_img), 
                          color=CLASS_COLORS[cls_id], thickness=1)
        else:
            if boxes_in:
                x_center = round(label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["width_idx"]] / 2.0)
                y_center = round(label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + label[DATA_ANN_FORMATS[ann_format]["height_idx"]] / 2.0)
            else:
                x_center = label[DATA_ANN_FORMATS[ann_format]["x_idx"]]
                y_center = label[DATA_ANN_FORMATS[ann_format]["y_idx"]]
            
            cv2.circle(img=img_arr, center=(x_center, y_center), radius=3, color=CLASS_COLORS[cls_id], thickness=-1)

    cv2.imwrite(f"{output_dir}/{img_name}_ann.jpg", img_arr)



def adjust_pseudo_dims_YOLO(data_dir: str, new_dims: dict, old_dims: dict, img_dims: dict, ann_format: str, max_overhang: float = 0.85) -> int: 
    """
    Change the dimensions of pseudo labels in a given directory containing YOLO training/validation data.
    Arguments:
        data_dir (str):             path to directory containing the annotations to be modified.
        new_dims (dict):            dictionary mapping class ids to the new box dimension.
        old_dims (dict):            dictionary mapping class ids to the old box dimension.
        ann_format (str):           format in which the annotations are provided
        max_box_overhang (float):   maximum percentage of a bounding box that can lie outside of an image without causing 
                                    the box to be excluded.
    Returns:
        int specifying the number of clipped/excluded boxes.                    
    """
    clipped_counter = 0

    for ann_file in Path(data_dir).glob("*.txt"):

        new_lines = []
        
        with open(ann_file, "r") as f:
            lines = [line.rstrip() for line in f]
            lines_floats = [[float(elem) for elem in l.split(" ")] for l in lines]

        for line in lines_floats:
            cat_id = int(line[MODEL_ANN_FORMATS[ann_format]["category_idx"]])
            xcenter = line[MODEL_ANN_FORMATS[ann_format]["center_x_idx"]]
            ycenter = line[MODEL_ANN_FORMATS[ann_format]["center_y_idx"]]
            old_with = line[MODEL_ANN_FORMATS[ann_format]["width_idx"]]
            old_height = line[MODEL_ANN_FORMATS[ann_format]["height_idx"]]
            old_width_abs =  old_with * img_dims["width"]
            old_height_abs =  old_height * img_dims["height"]
            
            # clipped box
            if old_width_abs < old_dims[cat_id]["width"]:
                """
                re-calculate the x coordinarte of the original center point
                to allow for re-clipping based on the new dimensions. 
                """ 
                residual = old_dims[cat_id]["width"] - old_width_abs
                xcenter_abs = xcenter * img_dims["width"]

                # check if clipped box exceeded the left or right border of the image
                exceeded_left = math.floor(xcenter_abs - (old_width_abs / 2)) == 0
                # if box was clipped from the left, the original center point lied to the left 
                sign = -1 if exceeded_left else 1 

                xcenter_adjusted = xcenter_abs + ((residual / 2) * sign)
                xcenter = xcenter_adjusted / img_dims["width"]

            # clipped box
            if old_height_abs < old_dims[cat_id]["height"]:
                """
                re-calculate the y coordinarte of the original center point
                to allow for re-clipping based on the new dimensions. 
                """ 
                residual = old_dims[cat_id]["height"] - old_height_abs
                ycenter_abs = ycenter * img_dims["height"]

                # check if clipped box exceeded the top or bottom of the image
                exceeded_top = math.floor(ycenter_abs - (old_height_abs / 2)) == 0
                # if box was clipped from the above, the original center point lied above 
                sign = -1 if exceeded_top else 1 

                ycenter_adjusted = ycenter_abs + ((residual / 2) * sign)
                ycenter = ycenter_adjusted / img_dims["height"]
            

            new_width = new_dims[cat_id]["width"] / img_dims["width"]
            new_height = new_dims[cat_id]["height"] / img_dims["height"]

            new_xmin = xcenter - (new_width / 2)
            new_xmax = xcenter + (new_width / 2)
            new_ymin = ycenter - (new_height / 2)
            new_ymax = ycenter + (new_height / 2)

                            
            if new_xmax > 1.0:
                overhang = new_xmax - 1.0
                # check the fraction of the box that exceeds the patch
                overhang_box_frac = overhang / new_width
                if overhang_box_frac > max_overhang:
                    clipped_counter += 1
                    continue
                new_width -= overhang
                xcenter -= overhang / 2.0
                            
            if new_ymax > 1.0:
                overhang = new_ymax - 1.0
                # check the fraction of the box that exceeds the patch
                overhang_box_frac = overhang / new_height
                if overhang_box_frac > max_overhang:
                    clipped_counter += 1
                    continue
                new_height -= overhang
                ycenter -= overhang / 2.0
            
            if new_xmin < 0.0:
                overhang = abs(new_xmin)
                # check the fraction of the box that exceeds the patch
                overhang_box_frac = overhang / new_width
                if overhang_box_frac > max_overhang:
                    clipped_counter += 1
                    continue
                new_width -= overhang
                xcenter += overhang / 2.0
                
            if new_ymin < 0.0:
                overhang = abs(new_ymin)
                # check the fraction of the box that exceeds the patch
                overhang_box_frac = overhang / new_height
                if overhang_box_frac > max_overhang:
                    clipped_counter += 1
                    continue
                new_height -= overhang
                ycenter += overhang / 2.0

            new_lines.append([cat_id, xcenter, ycenter, new_width, new_height])

        lines_str = [" ".join([str(number) for number in nl]) + "\n" for nl in new_lines]

        with open(ann_file, "w") as f:
            for line in lines_str:
                f.write(line)    

    return clipped_counter



def make_coco_file(imgs_dir: str, categories: list) -> str:
    """
    Generates a coco-dataset json to pass to the sahi-prediction function so that (non-visual) results are saved.
    Arguments: 
        imgs_dir (str):  path to the directory containing the images to be used for inference
        categories (list):  list containing dictionaries that map class ids to class names. 
    Returns: 
        string specifying the path to the created json file
    """
    coco_dict = {"images": [], "annotations": [], "categories": categories}

    id_count = 0

    for img in Path(imgs_dir).glob('*'):
        if img.suffix.upper() not in IMG_SUFFIXES:
            print(f"WARNING: skipping file {str(img)} as the file extension suggests that it's not a picture!")
            continue
        
        pil_img = Image.open(img)
        width, height = pil_img.size
        file_name = str(img)
        img_dict = {"width": width, "height": height, "file_name": file_name, "id": id_count}
        coco_dict["images"].append(img_dict)

        id_count += 1

    json_fn = f"{imgs_dir}/{Path(imgs_dir).stem}_coco_dataset.json"
    with open(json_fn, "w") as f:
        json.dump(coco_dict, f, indent=2)

    return json_fn