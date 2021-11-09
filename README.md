# DSA4266_Fish_Larva
This project aims to help do small objection detection. The current model implemented in this repository is YOLOV3 and YOLOV5.

# Project Structure
```bash
├── README.md
├── docker-compose.yml
├── sahi-yolov5.py
└── src
    ├── api_ui
    │   ├── Backend
    │   │   ├── Dockerfile
    │   │   ├── api.py
    │   │   ├── best.pt
    │   │   ├── prediction.py
    │   │   └── requirements.txt
    │   ├── Frontend
    │   │   ├── Dockerfile
    │   │   ├── frontend.py
    │   │   └── requirements.txt
    │   └── data
    │       ├── predict
    │       ├── predicted
    │       └── raw
    ├── data
    ├── image_tools.sh
    ├── requirements.txt
    ├── tools
    │   ├── image_concate.py
    │   ├── image_count.py
    │   ├── manual_data_aug.py
    │   ├── masking_unwanted.py
    │   └── tile_yolo.py
    ├── yolosliced
    │   ├── test.txt
    │   └── train.txt
    └── yolov3-master
      
```

# Getting Started
In this repository, we will be explaining the scripts that we have wrote to aid us in image processing/data augmentation. The following guide will help to explain on how to 
utilise these scripts. Scripts placed under tools are scripts used for inference when using YOLOV3. For our web app using our final model, the backend code can be found in Backend Folder while the frontend code can be found under Frontend Folder

# Backend

## Dockerfile
To initialise the image. To create the image and create the container, refer to the following.

```bash
docker build -t <image_name>:latest .

docker run -p 8005:8005 <image_name>
```

## best.pt
This is just the model used for YOLOV5. 

## prediction.py
Contains all the helper functions for inference. It is built solely around Sahi and YoloV5. 

## requirements.txt
Contains all the packages required to run the API and UI

## api.py
The api is created using a package in Python called FastAPI. It supports 3 functions - predict_image, download_results_json, download_image_file.
Python 3.5 or later is required for deployment. Initialise an Anaconda Environment with this Python version.

```bash
conda create --name fish-larva-demo python=3.5
conda activate fish-larva-demo
```

Install the required pacakges
```bash
pip install -r requirements.txt
```

To deploy the backend API, change to the following directory
```bash
cd src
cd api_ui
cd Backend
```

Run the application:
```bash
python api.py
```
To view API documentation, visit this link in your browser
```bash
http://localhost:8005/docs
```

# Frontend

## requirements.txt
Contains all the packages required to run the API and UI

## frontend.py
The frontend is created using a package is Python called Streamlit. To deploy the interface, let the backend running and open a new terminal. Change to the following directory
```bash
cd  ..
cd Frontend
```

Run the application
```bash
streamlit run frontend.py
```

To view the interface, visit the link in your terminal or the following link
```bash
http://localhost:8501/
```

# tools

## image_tools.sh (YOLOV3)
We created a pipeline that helps in inference of how the model predicts bounding boxes on the images. The variables will have to be edited to fit the environment that it is
running on. It utilises functions described below. 

### tile_yolo.py (YOLOV3)
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


### image_concate.py (YOLOV3)
The purpose of this script is to help paste back images. This is made specifically for test images. 
This scripts works in coordination with the output from tile_yolo.py. It will not be able to function properly otherwise.

```bash
--image_path            Directory path to data
--image_prefix          Image Label e.g. 20210903_09054 
--final_results_path    Directory path to store the pasted images
```

### image_count.py (YOLOV3)
The purpose of this script is to get the overall counts of each class. This is created because images are sliced and the counts received from calling YOLOV3 detect.py is for each sliced image. 

```bash
--image_labels_path     Directory path to data
--image_label_prefix    Image Label e.g. 20210903_09054 
--final_results_path    Directory path to store the txt file of all counts
```

### manual_data_aug.py (YOLOV3)
The purpose of this script is to get the data augmentation image and labels for the training dataset. This is created to increase the training dataset in hopes to reduce overfitting. 

```bash
--random_number         Random number from 1-3
--image_str             Image Label e.g. 20210729_131410_0_2.jpg
--label_str             Label name e.g. 20210729_131410_0_2.txt
--img_folder_name       Directory path to obtain images 
--lab_folder_name       Directory path to obtain labels
--image_folder_save     Directory path to store new images
--lab_folder_save       Directory path to store new labels
```

### masking_unwanted.py (YOLOV3)
The purpose of this script is to get remove the sections of the image which the group labelled as 4 to be masked out. This is created so that these areas do not interfere with the training of the dataset.

```bash
--image_str             Image Label e.g. 20210729_131410_0_2.jpg
--label_str             Label name e.g. 20210729_131410_0_2.txt
--img_folder_name       Directory path to obtain images 
--lab_folder_name       Directory path to obtain labels
--image_folder_save     Directory path to store new images
--lab_folder_save       Directory path to store new labels
```

