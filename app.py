import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np

@st.cache
def load_image(file):
    # st.image(Image.open(uploaded))
    img = Image.open(file)
    img = img.resize((300,300))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    return img

## Predict model function
def predict(image):
    model = load_model('weight_file/defect.h5')
    st.write(model.summary())


def main():
    st.header("Flaw Detection")
    
    st.markdown("This is the Defect detection program for classify front's flaw from images")
    with st.beta_expander(label="Show image sample"):
        choose_sample = st.selectbox('Select sample part', ["Not show sample","ok sample","NG sample"])
        if choose_sample == "ok sample":
            st.image('util/cast_ok_0_8.jpeg')
        elif choose_sample == "NG sample":
            col1, col2 = st.beta_columns(2)
            col1.image('util/cast_def_0_68.jpeg')
            col2.image('util/cast_def_0_355.jpeg')
    uploaded = st.file_uploader("Upload image",type=['png','jpeg','jpg'])
    if uploaded is not None:
        st.image(Image.open(uploaded))
        img = load_image(uploaded)
        st.write(img.shape)

if __name__=='__main__':
    main()

