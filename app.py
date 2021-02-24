import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np
import os

@st.cache
def load_image(file):
    # st.image(Image.open(uploaded))
    img = Image.open(file)
    img = img.resize((300,300))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    return img

def main():
    st.header("Flaw Detection")
    st.markdown("This is the Defect detection program for classify front's flaw from images")
    with st.beta_expander(label="Show image sample"):
        st.image('rose.jpg')
    uploaded = st.file_uploader("Upload image",type=['png','jpeg','jpg'])
    if uploaded is not None:
        st.image(Image.open(uploaded))
        img = load_image(uploaded)
        st.write(img.shape)

if __name__=='__main__':
    main()

