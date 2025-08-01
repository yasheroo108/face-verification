import streamlit as st
from deepface import DeepFace
import json
import numpy as np
import cv2

st.markdown("""
    <style>   
    div.stButton > button {
        font-size: 20px;
        padding: 1em 2em;
    }
    .stFileUploader {
        font-size: 18px;
    }
    .stBadge {
        font-size: 18px;
    }
    .st-emotion-cache-1kyxreq {
        font-size: 20px !important;     /* Increase font size */
        padding: 10px 20px !important;  /* Increase padding */
        border-radius: 12px !important; /* Round the corners more */
    }
    </style>
""", unsafe_allow_html=True)


st.title("Face Verification")
st.markdown("""
<p style='font-size:16px;'>
  Made by Yash Trivedi, powered by <a href='https://streamlit.io' target='_blank'>Streamlit</a> and 
  <a href='https://github.com/serengil/deepface' target='_blank'>DeepFace</a>.
</p>
""", unsafe_allow_html=True)


def to_opencv_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
person_one = st.file_uploader("Upload an image of the first person", type=["png", "jpg", "jpeg", "webp"])

if person_one is not None:
    st.image(person_one)

person_two = st.file_uploader("Upload an image of the second person", type=["png", "jpg", "jpeg", "webp"])

if person_two is not None:
    st.image(person_two)

verification_btn = st.button("Verify")

with st.spinner("Verifying..."):

    try:
        if verification_btn:
            if person_one is not None and person_two is not None:
                img1 = to_opencv_image(person_one)
                img2 = to_opencv_image(person_two)
                result = DeepFace.verify(img1_path = img1, img2_path = img2)
                result = (json.dumps(result, indent=2))

                if '"verified": true' in result:
                    st.info("Both faces belong to the same person")
                    st.code(json.dumps(result, indent=2))

                if '"verified": false' in result:
                    st.info("Both faces do not belong to the same person")
                    st.code(json.dumps(result, indent=2))


            else:
                st.warning("Please input both files")

    except ValueError as e:
        error_message = str(e)

        if "img1" in error_message:
            st.warning("No face detected in your first image")
        elif "img2" in error_message:
            st.warning("No face detected in your second image")
        else:
            st.warning("Unknown error occurred, please try with different images or later.")
            st.write(error_message)