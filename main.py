import io 
import time 

import streamlit as st 
import pypdfium2 
from PIL import Image

from agent import load_models, get_ocr_predictions, get_json_result


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


st.set_page_config(layout="wide")

models = load_models()

st.title("""
受領した請求書を自動で電子化 (Demo)
""")

col1, _, col2 = st.columns([.45, 0.1, .45])

in_file = st.sidebar.file_uploader(
    "PDFファイルまたは画像:", 
    type=["pdf", "png", "jpg", "jpeg", "gif", "webp"],
    )

if in_file is None:
    st.stop()

filetype = in_file.type
whole_image = False
if "pdf" in filetype:
    page_count = page_count(in_file)
    page_number = st.sidebar.number_input(f"ページ番号 {page_count}:", min_value=1, value=1, max_value=page_count)

    pil_image = get_page_image(in_file, page_number)
else:
    pil_image = Image.open(in_file).convert("RGB")

text_rec = st.sidebar.button("認識開始")

if pil_image is None:
    st.stop()
    
with col1:
    st.write("## アップロードされたファイル")
    st.image(pil_image, caption="アップロードされたファイル", use_column_width=True)
    
if text_rec:
    with col2:
        st.write("## 結果")
        
        # Placeholder for status indicator
        status_placeholder = st.empty()
        
        with st.spinner('現在ファイルを解析中です'):
            # Simulate model running time
            # time.sleep(5)  # Replace this with actual model running code
            predictions = get_ocr_predictions(pil_image, models)
            
            # Simulate OCR result as a JSON object
            json_predictions = get_json_result(predictions)
            
            # After model finishes
            status_placeholder.success('ファイルの解析が完了しました!')
            
            # Display the result
            st.write("解析後の内容:")
            st.json(json_predictions)
            # st.write(predictions)
    
