import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.title("Image Classification with Google's Teachable Machine Web Tool")
st.header("Brain Tumor Detection from MRI Images")
st.text("Upload a brain MRI Image for the model to tell if the Brain MRI is healthy or Not")

from img_classification import teachable_machine_classification


uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")

if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'keras_model.h5')
        if label == 1:
            st.write("The MRI scan has a brain tumor")
        else:
            st.write("The MRI scan is healthy")

