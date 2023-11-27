import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2, io
import easyocr

def sharpen_image(image_array):
    img = image_array
    
    mask = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]
    ])

    sharpened = cv2.filter2D(img, -1, mask)

    result = Image.fromarray(sharpened)

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Sharpened image.', use_column_width=True)

def smoothen_image(image_array):
    img = image_array
    
    mask = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]
    ])

    sharpened = cv2.filter2D(img, -1, mask)

    result = Image.fromarray(sharpened)

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Sharpened image.', use_column_width=True)


def text_recognition(image):
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)

    table_data = []
    for i in range(len(result)):
        r = []
        for j in range(len(result[i])):
            if j>0:
                r.append(result[i][j])
        table_data.append(r)

    df = pd.DataFrame(table_data, columns = ["Extracted Text","Confidence Level"])
    st.write(df)

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download as csv file.",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

st.sidebar.title("Image Processing Tool")
st.write("")

uploaded_file = st.sidebar.file_uploader("Upload an image to get started.", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    image_array = np.array(image)
    
    if st.sidebar.button("Sharpen Image", type="primary"):
        sharpen_image(image_array)

    if st.sidebar.button("Smoothen Image", type = "primary"):
        smoothen_image(image_array)

    if st.sidebar.button("Text Extraction", type = "primary"):
        text_recognition(image_array)