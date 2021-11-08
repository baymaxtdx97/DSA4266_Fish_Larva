from starlette.responses import FileResponse
import uvicorn
import json
import base64
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from prediction import load_model, predict, expected_yolo_format, table_summary



app = FastAPI()
model = load_model()

@app.post('/predict')
def predict_image(file: UploadFile= File(...)):
    # create path to store input file
    file_loc = './data/raw'
    # get file name without extension
    input_filename = file.filename[:-4]
    # store input file into that path
    with open((os.path.join(file_loc, file.filename).replace("\\", "/")), "wb+") as fileobject:
        fileobject.write(file.file.read())
    # read image stored in input path 
    image = os.path.join(file_loc, file.filename).replace("\\", "/")
    # create prediction using image and model
    predictions = predict(image, model)
    # create output json file 
    output_file_in = predictions.to_coco_annotations()
    output_file = expected_yolo_format(image, output_file_in)
    # create dataframe summary
    table_count_df = table_summary(output_file_in)

    # create an output directory based on image name
    output_loc = './data/predicted'
    # export prediction image to folder
    predictions.export_visuals(output_loc)
    # export json file to folder
    with open((os.path.join(output_loc, input_filename + ".json").replace("\\", "/")), 'w') as outfile:
        json.dump(output_file, outfile)
    # export dataframe summary as csv file 
    result = table_count_df.to_json()
    return {"file_name": input_filename,
            "count_df": result}

@app.get('/download_image_file')
def download_results_json():
    first_input = './data/predicted/prediction_visual.png'
    with open(first_input, 'rb') as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        json_b64 = json.dumps(b64)
        return {
            "json_b64": json_b64
        }

@app.get('/download_results_json')
def download_image_file(input_filename: str):
    first_input = os.path.join('./data/predicted', input_filename + ".json").replace('\\', "/")
    with open(first_input, 'rb') as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        json_b64 = json.dumps(b64)
        return {
            "json_b64": json_b64
        }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port = 8005)