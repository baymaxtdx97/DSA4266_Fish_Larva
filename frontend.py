import json
import os
import base64
import pandas as pd
import requests
import numpy as np
import streamlit as st
from PIL import Image
from streamlit.elements import media
from requests_toolbelt import MultipartEncoder


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{bin_str}" download="{os.path.basename(bin_file)}"><input type="button" value="Download"></a>'
    return href

#def load_image(image_file):
#	img = Image.open(image_file)
#	return img

def process(file, server_url:str):
    media = MultipartEncoder(
        fields = {
            "file":(
                file.name, 
                file.read(), 
                file.type)})
    req = requests.post(
        server_url, data = media, headers = {
            "Content-Type": media.content_type}, timeout=8000)
    return req

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(page_title = 'Fish Larva Object Detection', layout = 'wide')
st.title("Fish Larva Object Detection")

upload = st.container()
download = st.container()
display = st.container()
count = st.container()

with upload: 
    st.header("Upload an image and generate predictions")
    with st.expander("Upload an image in jpeg format"):
        image = st.file_uploader(
            "Choose an image", type =[
                'jpg', 'jpeg'])
        if image is not None:
            if st.button("Run Prediction"):
                res = process(image, f"http://localhost:8005/predict")
                st.session_state.image_name_uploaded = res.json().get('file_name')
                json_result_df = res.json().get('count_df')
                dataframe = pd.read_json(json_result_df)
                st.session_state.dataframe_count = dataframe
                st.write("Prediction is completed. Please proceed to download your file and view output")

with download:
    with st.expander("Download your file"):
        if st.button("Click here to continue"):
            json_response = requests.get(
                f"http://localhost:8005/download_results_json", 
                params ={
                    "input_filename": st.session_state.image_name_uploaded})
            json_b64 = json_response.json().get('json_b64')
            b64 = json.loads(json_b64)
            bin_file = st.session_state.image_name_uploaded +'.json'
            href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(bin_file)}"><input type="button" value="Download"></a>'
            st.markdown(href, unsafe_allow_html= True)

with display:
    with st.expander("Download your annotated image"):
        if st.button("Click here to proceed"):
            json_response = requests.get(
                f"http://localhost:8005/download_image_file")
            json_b64 = json_response.json().get('json_b64')
            b64 = json.loads(json_b64)
            bin_file = st.session_state.image_name_uploaded +'.png'
            href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(bin_file)}"><input type="button" value="Download"></a>'
            st.markdown(href, unsafe_allow_html= True)

with count:
    with st.expander("View Counts for classes"):
        if st.button("View"):
            output_df = st.session_state.dataframe_count
            st.dataframe(output_df)
            





