from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction

import json

# Edit to where ever we store best.pt file in our docker container
yolov5_model = './best.pt'

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

    object_prediction_list = result.to_coco_annotations()

    with open(path_to_save_results, 'w') as outfile:
        json.dump(object_prediction_list, outfile)




if __name__ == "__main__":
    main()