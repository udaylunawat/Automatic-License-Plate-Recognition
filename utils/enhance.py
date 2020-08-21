import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageEnhance

crop, image = None, None
img_size, crop_size = 600, 400

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def cropped_image(image, b):
    crop = cv2.rectangle(np.array(image), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    crop = crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    crop = Image.fromarray(crop)

    return crop
    
# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def enhance_crop(crop):

    st.write("## Enhanced License Plate")

    slider = st.sidebar.empty()
    enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring","Cannize"])
    

    if enhance_type == 'Original':
        output_image = crop
        st.image(output_image, width=crop_size, caption = enhance_type)

    elif enhance_type == 'Gray-Scale':
        temp = np.array(crop.convert('RGB'))
        output_image = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
        st.image(output_image, width = crop_size, caption = enhance_type)

    elif enhance_type == 'Contrast':
        c_rate = slider.slider("Contrast",0.2,8.0,(3.5))
        enhancer = ImageEnhance.Contrast(crop)
        output_image = enhancer.enhance(c_rate)
        st.image(output_image,width = crop_size, caption = enhance_type)

    elif enhance_type == 'Brightness':
        c_rate = slider.slider("Brightness",0.2,8.0,(1.5))
        enhancer = ImageEnhance.Brightness(crop)
        output_image = enhancer.enhance(c_rate)
        st.image(output_image, width = crop_size, caption = enhance_type)

    elif enhance_type == 'Blurring':
        temp = np.array(crop.convert('RGB'))
        blur_rate = slider.slider("Blur",0.2,8.0,(1.5))
        img = cv2.cvtColor(temp,1)
        output_image = cv2.GaussianBlur(img,(11,11),blur_rate)
        st.image(output_image, width = crop_size, caption = enhance_type)
    
    elif enhance_type == 'Cannize':
        output_image = cannize_image(crop)
        st.image(output_image, width = crop_size, caption = enhance_type)