from PIL import Image
import ast
import os 

import streamlit as st 
from surya.ocr import run_ocr
from surya.model.detection import model
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import openai


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


LANGUAGES = ["ja", "en"]

@st.cache_resource()
def load_models():
    det_model, det_processor = model.load_model(), model.load_processor()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    return det_processor, det_model, rec_model, rec_processor

def get_ocr_predictions(pil_image, models):
    det_processor, det_model, rec_model, rec_processor = models
    img_pred = run_ocr([pil_image], [LANGUAGES], det_model, det_processor, rec_model, rec_processor)[0]
    predictions = []
    for line in img_pred.text_lines:
        predictions.append([ast.literal_eval(str(line.bbox)), str(line.text)])
    print(predictions)
    return predictions

def get_json_result(predictions):
    user_prompt = f"""
    ### Instruction:
    You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object. 
    Don't make up value not in the Input. Output must be a well-formed JSON object. And both key and values in the json object should be in Japanese or Number. Don't lose any information. 
    ```json

    ### Input:
    {predictions}
    
    ### Output:
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": user_prompt}
        ]
    )
    json_result = response.choices[0].message.content
    return json_result