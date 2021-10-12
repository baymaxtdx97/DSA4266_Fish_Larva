# DSA4266_Fish_Larva
This project aims to help do small objection detection. The current model implemented in this repository is YOLOV3.

## Project Structure
```bash
 src
    ├── data
    ├── image_tools.sh
    ├── tools
    │   ├── image_concate.py
    │   ├── image_count.py
    │   ├── masking.py
    │   ├── process_labels.py
    │   └── tile_yolo.py
    ├── yolosliced
    └── yolov3-master
      
```

## Getting Started
In this repository, we will be explaining the scripts that we have wrote to aid us in image processing/data augmentation. The following guide will help to explain on how to 
utilise these scripts. All scripts are placed under the folder tools. 


## image_tools.sh
We created a pipeline that helps in inference of how the model predicts bounding boxes on the images. The variables will have to be edited to fit the environment that it is
running on. It utilises functions described below. 

### tile_yolo.py
This script has been created by Rostyslav Neskorozhenyi and the original repository can be found [here](https://github.com/slanj/yolo-tiling).
Purpose of this script is to help us slice our images into a desired resolution.
For our project, We ran the script with the following configurations. When running the script for train images, there has to be an additional script
classes.names that contains the classes. 

```bash
# This is if you are trying to slice test images with no txt files for bounding boxes
-source         Directory path to data. Needs to contain both images and labels
-falsefolder    Directory path to store images where there are no bounding boxes
-ext .jpg       Format you want to save the images
-size           Size of Images

# Add this parameters if you are trying to slice training images
-target         Directory to save images and resized bounding boxes txt files
```


### image_concate.py
The purpose of this script is to help paste back images. This is made specifically for test images. 
This scripts works in coordination with the output from tile_yolo.py. It will not be able to function properly otherwise.

```bash
--image_path            Directory path to data
--image_prefix          Image Label e.g. 20210903_09054 
--final_results_path    Directory path to store the pasted images
```

### image_count.py
The purpose of this script is to get the overall counts of each class. This is created because images are sliced and the counts received
from calling YOLOV3 detect.py is for each sliced image. 

```bash
--image_labels_path     Directory path to data
--image_label_prefix    Image Label e.g. 20210903_09054 
--final_results_path    Directory path to store the txt file of all counts
```