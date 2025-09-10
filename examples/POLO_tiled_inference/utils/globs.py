DATA_ANN_FORMATS = {    
    "BX_WH": {"label_key": "bbox", "x_min_idx": 0, "y_min_idx": 1, "width_idx": 2, "height_idx": 3, "category_key": "category_id"},
    "PT_DEFAULT": {"label_key": "point", "x_idx": 0, "y_idx": 1, "category_key": "category_id"}
}


MODEL_ANN_FORMATS = {
    "YOLO_BX": {"center_x_idx": 1, "center_y_idx": 2, "width_idx": 3, "height_idx": 4, "category_idx": 0},
    "YOLO_PT":  {"x_idx": 2, "y_idx": 3, "category_idx": 0, "radius_idx": 1}
}

PATCH_XSTART_IDX = 0
PATCH_YSTART_IDX = 1

# BGR colors
CLASS_COLORS = {0: (134, 22, 171),      # purple
                1: (38, 154, 247),      # brown
                2: (204, 232, 19),      # turquoise
                3: (0, 97, 242),        # orange
                4: (0, 255, 0),         # green
                5: (255, 0, 0)}         # blue


NAMES_DIRECTORIES_FILES = {"count_directory":       "image_counts",
                           "train":                 "train",
                           "val":                   "val",
                           "test":                  "test",
                           "val_full_dir":          "val_full_imgs",
                           "inference_output_dir":  "detections",
                           "annotations_json":      "annotations.json",
                           "ultralytics_yaml":      "dataset.yaml",
                           "dataset_file_SAHI":     "dataset_SAHI.json",
                           "total_counts_file":     "counts_total.json",
                           "FL_all_data_dir":       "all_data",
                           "FL_client_stats_json":  "client_stats.json",
                           "FL_state_dict":         "state.pt"
                          }