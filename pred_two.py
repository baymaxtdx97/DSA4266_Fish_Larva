import cv2
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
from typing import Dict, List, Union

import base64
import json


def base_64_img(img_path: str) -> str:
    with open(img_path, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string

def get_img_shape(img_path: str) -> List[int]:
    img = cv2.imread(img_path)
    return img.shape

def convert_labels(size: List[int], x1: int, y1:int , x2:int , y2:int) -> List[float]:
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
    
    new_mid_x = (1/dh) - y
    new_mid_y = x
    new_width_x = h
    new_width_y = w
    
    # normalize
    x = new_mid_x*dh
    w = new_width_x*dh
    y = new_mid_y*dw
    h = new_width_y*dw
    
    
    return [x, y, w, h]

def expected_yolo_format(img_path: str, coco_annotations: List[Dict[str, 
                        Union[float, int, List[float]]]] ) -> List[Dict[str, Union[str, List[Dict[str, Union[int, float, List[float]]]]]]]:
    
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



# Edit to where ever we store best.pt file in our docker container
yolov5_model = './models/best.pt'

def main(path_to_image:str, path_to_save_image:str, path_to_save_results:str) -> None:
    """
    :param path_to_image: Directory to find Image
    :param_to_save_results: Directory to save json file
    :return: Returns None but will save the Image from Inference and Json File
    """
    # Initialise YoloV5 with Sahi
    detection_model = Yolov5DetectionModel(
    model_path=yolov5_model,
    confidence_threshold=0.4, # This the threshold you set to see where is the sweet spot for inference
    device="cpu", # or 'cuda'
    )

    # Get Inference Results
    result = get_sliced_prediction(
    path_to_image,
    detection_model,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
    )

    result.export_visuals(export_dir = path_to_save_image)

    object_prediction_list = result.to_coco_predictions()

    json_output = expected_yolo_format(path_to_image, object_prediction_list)


    with open(path_to_save_results, 'w') as outfile:
        json.dump(json_output, outfile)

base_64_img('C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/raw/20210729_131410.jpg')
#main('C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/raw/20210729_131410.jpg',
#'C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/predicted',
#'C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/predicted/output.json')


#if __name__ == "__main__":
#    main()