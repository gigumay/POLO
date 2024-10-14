# POLO - Point-based multi-class detectiom 
### Polo is a multi-class object detection framework that outputs and trains on point labels. The architecture largely corresponds to that of the YOLOv8 model developed by [ultralytics](https://www.ultralytics.com), which can be found on the company's official [GitHub](https://github.com/ultralytics/ultralytics/tree/main/ultralytics). Beyond the code in this repo, you can refer to [the associated research paper](https://www.google.com) for details on the changes made to the original YOLOv8 architecture.


## Installation
POLO can be downloaded and installed using pip like any other python package:

`pip install git+https://github.com/gigumay/POLO.git#egg=ultralytics`

Since the POLO package is a fork of the ultralytics library, the model can be imported by adding 

`from ultralytics import YOLO`

to your script. This will not only give you access to POLO, but also to all other YOLO models and functionalities (YOLOv3-8, pose-estimation, segmentation, etc.). For instructions on anything that extends beyond POLO, please consult the official [ultralytics documentation](https://docs.ultralytics.com). Note that POLO was developed in python 3.10, which therefore is the version we reccomend using with the environment POLO is installed into. 

## Training POLO
### Data-prerocessing 
Datsets should be divided into a training- and validation-set, as well as the corresponding images should be stored in separate folders. Furthermore, like is the case for YOLOv8, the architecture of POLO is designed to take input images of size 640x640 pixels. If you are working with images that exceed this size, it can be beneficial to split them into patches that match the dimensions expected by POLO. While the framework will accept arbitrary sized inputs, large images will have to be downsampled to fit the architecture, which will reduce the resolution and may cause the loss of important features/details. 
Then, for each image (or image-patch) in the training- and validation-set, the objects it contains must be specified as ground truth labels in a `.txt`-file. The `.txt`-files must be located in the same folder as the corresponding images, and the file-names must be identical (except, of course, for the file extension/format). In the `.txt`-file each object must be defined as a point in a separate line, and every line must be formatted as follows:

`class_id radius x_rel y_rel`

1. `class_id`: An integer number indicating the id of the object class.
2. `radius`: The radius (in pixels) defiend for the object class (please refer to [the original POLO paper](https://www.google.com) for details)
3. `x_rel`: The relative x-coordinate of a point lying on the object (ideally its center-point).
4. `y_rel`: The relative y-coordinate of a point lying on the object (ideally its center-point).

As can be understood, the location of an point label, must be passed as its position relative to the image dimensions. These relative coordinates can be obtained by dividing the absolute coordinates - i.e., the pixel-row and -column - by the image-width/-height. Please see the below graphic for a visual representation of this concept. Note that if an image does not contain any objects, an empty `.txt` file must be generated.

![Image](./README_imgs/coords.png)

Finally, a `.yaml` file must be created to specify the path to the training- and validation-data, as well as the IDs, names, and radii of the object classes. This file must be structured as follows: 

```
path: /path/to/project/main_folder
train: splits/train
val: splits/val

names:
    0: name0
    1: name1
    ...

radii:
    0: radius_class_0
    1: radius_class_1
    ...
```

Importantly, the path specified after the `path` keyword must contain the path to main folder of the project, i.e., the directory that normally contains the training- and -validation folder. The paths after the `train` and `val` keywords must be relative to that main folder. Finally, The `names` keyword contains a mapping from numerical class IDs (integers; same IDs that go into the `.txt` annotation files) to class names (strings; e.g., "dog"), whereas the radii for all classes must be passed under the `radii` keyword.  


To summarize, this is what a project structure matching the above explanations would look like:
```
main_folder
|   
+-- dataset.yaml
|
+-- splits
    |
    +-- train
        |
        +-- img0_train.jpg
        +-- img0_train.txt
        +-- img1_train.jpg
        +-- img1_train.txt
        ...
    +-- val
        |
        +-- img0_val.jpg
        +-- img0_val.txt
        +-- img1_val.jpg
        +-- img1_val.txt

```

### Loading the model and initiating training





- tune DoR and radii
