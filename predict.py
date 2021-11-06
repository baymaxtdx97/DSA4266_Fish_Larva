from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction

import json

# Edit to where ever we store best.pt file in our docker container

def main(path_to_image:str, path_to_save_image:str, path_to_save_results:str, path_to_yolo: str):
    """
    :param path_to_image: Directory to find Image
    :param path_to_save_image: Directory to save image
    :param_to_save_results: Directory to save json file
    :return: Returns None but will save the Image from Inference and Json File
    """
    # Initialise YoloV5 with Sahi
    detection_model = Yolov5DetectionModel(
    model_path=path_to_yolo,
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

    object_prediction_list = result.to_coco_annotations()

    #with open(path_to_save_results, 'w') as outfile:
    #    json.dump(object_prediction_list, outfile)

main('C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/raw/20210903_100734.jpg',
'C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/predicted',
'C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/data/predicted', 
'C:/Users/Arushi Gupta/Documents/Y4S1/DSA4266_Fish_Larva/models/best.pt')

#if __name__ == "__main__":
#    main('../../../../data/raw/20210903_100734.jpg','../../../../data/predicted', '../../../../data/predicted')

