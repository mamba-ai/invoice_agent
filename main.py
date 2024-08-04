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

in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])

if in_file is None:
    st.stop()

filetype = in_file.type
whole_image = False
if "pdf" in filetype:
    page_count = page_count(in_file)
    page_number = st.sidebar.number_input(f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count)

    pil_image = get_page_image(in_file, page_number)
else:
    pil_image = Image.open(in_file).convert("RGB")

text_rec = st.sidebar.button("Run OCR")

if pil_image is None:
    st.stop()
    
with col1:
    st.write("## Uploaded Image")
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)
    
if text_rec:
    with col2:
        st.write("## Results")
        
        # Placeholder for status indicator
        status_placeholder = st.empty()
        
        with st.spinner('Model is running...'):
            # Simulate model running time
            # time.sleep(5)  # Replace this with actual model running code
            predictions = get_ocr_predictions(pil_image, models)
            
            # Simulate OCR result as a JSON object
            json_predictions = get_json_result(predictions)
            
            # After model finishes
            status_placeholder.success('Model has finished running!')
            
            # Display the result
            st.write("OCR Result:")
            st.json(json_predictions)
            # st.write(predictions)
    
