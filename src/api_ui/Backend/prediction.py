import cv2
import pandas as pd


from PIL import Image
from io import BytesIO
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
from typing import Dict, List, Union

import base64
import json
from collections import Counter

def base_64_img(img_path: str) -> str:
    """
    :param img_path: path of image
    :return: returns base64 image in utf-8 format
    """
    with open(img_path, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
        base64_string = my_string.decode('utf-8')
    return base64_string

def get_img_shape(img_path: str) -> List[int]:
    """
    :param img_path: path of image
    :return: shape of image
    """
    img = cv2.imread(img_path)
    print(img.shape)
    return img.shape

def convert_labels(size: List[int], x1: int, y1:int , x2:int , y2:int) -> List[float]:
    """
    This only works with sahi because rotation happens in sahi and thus this script is configured to deal with the rotation as well.
    :param size: size of image
    :param x1: coordinate of bounding box in coco format
    :param x2: coordinate of bounding box in coco format
    :param y1: coordinate of bounding box in coco format
    :param y2: coordinate of bounding box in coco format
    :return: list of coordinates in yolo format
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin

    # if image is in portrait mood
    # do a roation
    if size[0] > size[1]:
        dw = 1./size[0]
        dh = 1./size[1]
        new_mid_x = (1/dh) - y
        new_mid_y = x
        new_width_x = h
        new_width_y = w
    
        # normalize
        x = new_mid_x*dh
        w = new_width_x*dh
        y = new_mid_y*dw
        h = new_width_y*dw
    else:
        new_mid_x = (1/dw) - x
        new_mid_y = (1/dh) - y
        new_width_x = w
        new_width_y = h

        #normalize
        x = new_mid_x*dw
        w = new_width_x*dw
        y = new_mid_y*dh
        h = new_width_y*dh
    
    return [x, y, w, h]

def expected_yolo_format(img_path: str, coco_annotations: List[Dict[str, 
                        Union[float, int, List[float]]]] ) -> List[Dict[str, Union[str, List[Dict[str, Union[int, float, List[float]]]]]]]:
    """
    converts coco format to the expected format for assignment
    :param img_path: path of image
    :coco_annotations: inference output in coco format
    :return: returns the expected json format
    """
    # To get the image name, e.g 2021_093203. Remember to change the split based on how the path is configured
    img_name = img_path.split('/')[-1][:-4]

    # Get img size
    size = get_img_shape(img_path)

    #Get base64 img
    base64_img = base_64_img(img_path)

    prediction_list = []
    for labels in coco_annotations:
        bounding_box = convert_labels(size, labels['bbox'][0], labels['bbox'][1], 
                            labels['bbox'][2] + labels['bbox'][0], labels['bbox'][3] + labels['bbox'][1])
    
        prediction_list.append({'predicted_class': labels['category_id'],
                            'confidence': labels['score'],
                            'bounding_box': bounding_box
                           })

    return {'filename': img_name, 'image_base64': base64_img, 'predictions': prediction_list}


def load_model():
    """
    loeads the yolov5 model in sahi format
    """
    path_to_yolo = 'best.pt'
    detection_model = Yolov5DetectionModel(
    model_path=path_to_yolo,
    confidence_threshold=0.4, # This the threshold you set to see where is the sweet spot for inference
    device="cpu", # or 'cuda'
    )
    return detection_model

def predict(image: str, model_used):
    """
    :param image: path of image
    :param model: model used for inference, only works with sahi yolov5 or other mmdetection model in sahi
    :return: inference results
    """
    result = get_sliced_prediction(
    image,
    model_used,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
    )
    return result

def table_summary(predicted_label: List[Dict[str, Union[float, int, List[float]]]]) -> pd.DataFrame:
    """
    :param predicted_label: output json format of sahi inference
    :return: returns dataframe containing summary count of each class label
    """
    counter_list = Counter([label['category_name'] for label in predicted_label])
    table_summary = pd.DataFrame.from_records(counter_list.most_common(), columns=['Label','count'])
    return table_summary

#get_img_shape('./data/raw/20210903_095054.jpg')