from PIL import Image
import ast
import os 

import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.hyperlink import Hyperlink

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
    
    
def json_to_excel_with_links(json_data, excel_file_path):
    """
    Convert JSON data to Excel file with nested JSONs in new sheets and hyperlinks.
    
    Parameters:
    json_data (str or dict): JSON data as a string or dictionary.
    excel_file_path (str): Path where the Excel file will be saved.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    
    wb = Workbook()
    main_sheet = wb.active
    main_sheet.title = 'Main'

    def add_sheet(sheet_name, data):
        """
        Add a new sheet with given data.
        
        Parameters:
        sheet_name (str): The name of the new sheet.
        data (list or dict): The data to be added to the new sheet.
        """
        df = pd.DataFrame(data)
        # if isinstance(data, list):
            # df.insert(0, 'Sequence', range(1, len(df) + 1))
        
        new_sheet = wb.create_sheet(title=sheet_name)
        for r in dataframe_to_rows(df, index=False, header=True):
            new_sheet.append(r)

    def normalize_json(json_obj, sheet_name_prefix=''):
        """
        Normalize JSON object to handle nested structures and add new sheets.
        
        Parameters:
        json_obj (dict or list): JSON object or list of JSON objects.
        sheet_name_prefix (str): Prefix for naming nested sheets.
        
        Returns:
        dict: Normalized JSON object.
        """
        if isinstance(json_obj, dict):
            normalized_dict = {}
            for k, v in json_obj.items():
                if isinstance(v, dict):
                    sheet_name = f"{sheet_name_prefix}{k}"
                    add_sheet(sheet_name, [v])
                    normalized_dict[k] = f'[{sheet_name}]'
                elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                    sheet_name = f"{sheet_name_prefix}{k}"
                    add_sheet(sheet_name, v)
                    normalized_dict[k] = f'[{sheet_name}]'
                else:
                    normalized_dict[k] = v
            return normalized_dict
        elif isinstance(json_obj, list):
            return [normalize_json(item, sheet_name_prefix) for item in json_obj]
        else:
            return json_obj

    normalized_data = []
    if isinstance(json_data, list):
        normalized_data = normalize_json(json_data)
    else:
        normalized_data.append(normalize_json(json_data))
    
    df_main = pd.DataFrame(normalized_data)
    for r in dataframe_to_rows(df_main, index=False, header=True):
        main_sheet.append(r)
    
    # Add hyperlinks to the main sheet
    for row in main_sheet.iter_rows(min_row=2, max_col=len(df_main.columns), max_row=main_sheet.max_row):
        for cell in row:
            if isinstance(cell.value, str) and cell.value.startswith('[') and cell.value.endswith(']'):
                sheet_name = cell.value.strip('[]')
                cell.hyperlink = f"#{sheet_name}!A1"
                cell.value = sheet_name

    wb.save(excel_file_path)
    
    
if __name__ == "__main__":
    json_data = '''
    [
        {"name": "John", "age": 30, "city": "New York"},
        {"name": "Anna", "age": 22, "city": "London"},
        {"name": "Mike", "age": 32, "city": "San Francisco"}
    ]
    '''
    json_data_2 = '''
        {
  "請求書": "御請求書",
  "会社情報": {
    "会社名": "株式会社SNSソフト",
    "郵便番号": "〒101-0031",
    "住所": "東京都千代田区東神田・TQ東神田ビル2階",
    "電話番号": "03-1234-5678"
  },
  "宛先": "株式会社ABC 御中",
  "挨拶文": "平素は格別のご高配に賜り、誠にありがとう御座います。",
  "担当": "担当",
  "請求内容": "下記の通りご請求申し上げます。",
  "合計金額": "¥550,000（消費税含）",
  "お支払期日": "2024年5月31日",
  "明細": [
    {
      "商品番号": "1",
      "商品名": "システム開発支援",
      "作業年月": "2024年4月",
      "数量": "1.0",
      "商品単価": "500,000",
      "金額": "500,000",
      "備考": ""
    },
    {
      "商品番号": "2",
      "商品名": "システム開発支援",
      "作業年月": "2024年4月",
      "数量": "1.0",
      "商品単価": "500,000",
      "金額": "500,000",
      "備考": ""
    }
  ],
  "小計": "500,000",
  "消費税": "50,000",
  "合計金額再掲": "550,000",
  "振込先": {
    "銀行名": "○○銀行 ××支店",
    "振込種別": "普通",
    "口座番号": "1234567",
    "口座名義": "株式会社SNSソフト"
  }
}
    '''

    # json_to_excel(json_data_2, 'output_2.xlsx')
    json_to_excel_with_links(json_data_2, 'output_with_links.xlsx')