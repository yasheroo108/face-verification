import streamlit as st
from deepface import DeepFace
import numpy as np
import cv2
import json
from PIL import Image, ImageOps
from io import BytesIO
import tempfile
import os
import gc


st.set_page_config(
    page_title="Face Verification",
    page_icon="🥸",
    layout="centered",
)

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
        font-size: 20px !important;
        padding: 10px 20px !important;
        border-radius: 12px !important;
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


def validate_and_resize_image(uploaded_file, max_size_mb=10, max_dimension=1024):
    """
    Validate, resize and convert image to prevent crashes
    """
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return None, f"File too large ({file_size_mb:.1f}MB). Please use images smaller than {max_size_mb}MB."
        
        uploaded_file.seek(0)
        
        pil_image = Image.open(uploaded_file)
        
        pil_image = ImageOps.exif_transpose(pil_image)
        
        if pil_image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if pil_image.width > max_dimension or pil_image.height > max_dimension:
            pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        return pil_image, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def pil_to_opencv(pil_image):
    """
    Convert PIL image to OpenCV format safely
    """
    try:
        img_array = np.array(pil_image)
        
        opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return opencv_image, None
    except Exception as e:
        return None, f"Error converting image: {str(e)}"


def save_temp_image(opencv_image):
    """
    Save opencv image to temporary file (DeepFace works better with file paths)
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        temp_file.close()
        
        cv2.imwrite(temp_path, opencv_image)
        
        return temp_path, None
    except Exception as e:
        return None, f"Error saving temporary image: {str(e)}"


def cleanup_temp_files(*file_paths):
    """
    Clean up temporary files and force garbage collection
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass
    
    gc.collect()


if 'verification_count' not in st.session_state:
    st.session_state.verification_count = 0

if st.session_state.verification_count > 5:
    st.warning("⚠️ Multiple verifications detected. If the app becomes slow, refresh the page to clear memory.")

person_one = st.file_uploader(
    "Upload an image of the first person", 
    type=["png", "jpg", "jpeg"], 
    key="person_one"
)

if person_one is not None:
    pil_img1, error1 = validate_and_resize_image(person_one)
    if error1:
        st.badge(f"First image error: {error1}", color="red")
        person_one = None
    else:
        st.image(pil_img1, caption="First Person", use_container_width=True, width=300)
        st.badge("Image processed successfully", color="green")

person_two = st.file_uploader(
    "Upload an image of the second person", 
    type=["png", "jpg", "jpeg"],  
    key="person_two"
)

if person_two is not None:
    pil_img2, error2 = validate_and_resize_image(person_two)
    if error2:
        st.badge(f"First image error: {error2}", color="red")
        person_two = None
    else:
        st.image(pil_img2, caption="Second Person", use_container_width=True, width=300)
        st.badge("Image processed successfully", color="green")

verification_btn = st.button(
    "🔍 Verify Faces", 
    disabled=(person_one is None or person_two is None),
    help="Both images must be uploaded successfully first"
)

if verification_btn:
    st.session_state.verification_count += 1
    
    with st.spinner("Verifying..."):
        temp_path1, temp_path2 = None, None
        
        try:
            if person_one is None or person_two is None:
                st.error("❌ Please upload both images successfully before verifying.")
            else:
                pil_img1, error1 = validate_and_resize_image(person_one)
                pil_img2, error2 = validate_and_resize_image(person_two)
                
                if error1 or error2:
                    st.error(f"❌ Image processing failed: {error1 or error2}")
                else:
                    cv_img1, cv_error1 = pil_to_opencv(pil_img1)
                    cv_img2, cv_error2 = pil_to_opencv(pil_img2)
                    
                    if cv_error1 or cv_error2:
                        st.error(f"❌ Image conversion failed: {cv_error1 or cv_error2}")
                    else:
                        temp_path1, temp_error1 = save_temp_image(cv_img1)
                        temp_path2, temp_error2 = save_temp_image(cv_img2)
                        
                        if temp_error1 or temp_error2:
                            st.error(f"❌ Temporary file creation failed: {temp_error1 or temp_error2}")
                        else:
                            try:
                                result = DeepFace.verify(
                                    img1_path=temp_path1, 
                                    img2_path=temp_path2,
                                    enforce_detection=False  
                                )
                                
                                cleanup_temp_files(temp_path1, temp_path2)
                                temp_path1, temp_path2 = None, None
                                
                                if result.get("verified", False):
                                    st.success("✅ **MATCH**: Both faces belong to the same person")
                                    st.caption("❗ Results may not always be accurate.")
                                    distance = result.get("distance", "N/A")
                                    threshold = result.get("threshold", "N/A")
                                    if distance != "N/A" and threshold != "N/A":
                                        confidence = (1 - distance/threshold) * 100 if threshold > 0 else 0
                                        st.info(f"🎯 Confidence: {confidence:.1f}% (Distance: {distance:.3f}, Threshold: {threshold:.3f})")
                                else:
                                    st.error("❌ **NO MATCH**: These faces belong to different people")
                                    st.caption("❗ Results may not always be accurate.")
                                    distance = result.get("distance", "N/A")
                                    threshold = result.get("threshold", "N/A")
                                    if distance != "N/A" and threshold != "N/A":
                                        st.info(f"📊 Distance: {distance:.3f}, Threshold: {threshold:.3f}")
                                
                                with st.expander("🔍 View Raw Result"):
                                    st.json(result)
                                
                            except ValueError as e:
                                error_msg = str(e).lower()
                                if "face could not be detected" in error_msg:
                                    st.error("❌ **Face Detection Failed**")
                                    st.info("💡 **Tips for better results:**")
                                    st.markdown("""
                                    - Ensure faces are clearly visible and well-lit
                                    - Face should be facing forward (not profile)
                                    - Remove sunglasses, masks, or face coverings
                                    - Use high-quality, non-blurry images
                                    - Try different photos if detection fails
                                    """)
                                else:
                                    st.error("❌ **Verification Error**")
                                    st.info("Please try with different images. If the problem persists, the images may not be suitable for face detection.")
                            
                            except Exception as e:
                                st.error("❌ **Unexpected Error**")
                                st.info("The app encountered an issue. Please refresh the page and try again with different images.")
                        
        except Exception as e:
            st.error("❌ **Critical Error**")
            st.info("Please refresh the page and try again. If using iPhone, make sure to use JPEG/PNG format only.")
            
        finally:
            cleanup_temp_files(temp_path1, temp_path2)
