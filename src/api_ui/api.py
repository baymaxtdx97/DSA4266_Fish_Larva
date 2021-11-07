from starlette.responses import FileResponse
import uvicorn
import json
import base64
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from prediction import read_image, load_model, predict, table_summary



app = FastAPI()
model = load_model()

@app.post('/predict')
def predict_image(file: UploadFile= File(...)):
    # create path to store input file
    file_loc = './data/raw'
    # get file name without extension
    input_filename = file.filename[:-4]
    # store input file into that path
    with open(os.path.join(file_loc, file.filename), "wb+") as fileobject:
        fileobject.write(file.file.read())
    # read image stored in input path 
    image = read_image(os.path.join(file_loc, file.filename).replace("\\", "/"))

    # create prediction using image and model
    predictions = predict(image, model)
    # create output json file 
    output_file = predictions.to_coco_annotations()
    # create dataframe summary
    table_count_df = table_summary(output_file)

    # create an output directory based on image name
    output_loc = './data/predicted'
    # export prediction image to folder
    predictions.export_visuals(output_loc)
    # export json file to folder
    with open(os.path.join(output_loc, input_filename + ".json"), 'w') as outfile:
        json.dump(output_file, outfile)
    # export dataframe summary as csv file 
    result = table_count_df.to_json()
    return {"file_name": input_filename,
            "count_df": result}
    #return FileResponse(os.path.join(output_loc, input_filename + ".json"), filename = input_filename + ".json")

@app.get('/getimage')
def get_annotated_image():
    return FileResponse(os.path.join('./data/predicted', 'prediction_visual.png'))

@app.get('/getjson')
def get_json(input_filename: str):
    #return FileResponse('./data/predicted/output.json', filename = 'output.json')
    first_input = os.path.join('./data/predicted', input_filename + ".json").replace('\\', "/")
    return FileResponse(first_input, filename = input_filename + ".json")

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