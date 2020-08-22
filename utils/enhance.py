import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageEnhance
import re

crop, image = None, None
img_size, crop_size = 600, 400


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#skew correction
def deskew(image):
  coords = np.column_stack(np.where(image > 0))
  angle = cv2.minAreaRect(coords)[-1]
  if angle < -45:
    angle = -(90 + angle)
  else:
      angle = -angle
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
  return rotated

def cannize_image(image):
    new_img = np.array(image.convert('RGB'))
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
    rgb = np.array(crop.convert('RGB'))
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    slider = st.sidebar.empty()
    enhance_type = st.sidebar.radio("Enhance Type",\
                                   ["Original","Gray-Scale","Contrast",\
                                    "Brightness","Blurring","Cannize",\
                                    "Remove_noise", "Thresholding", "Dilate",\
                                    "Opening","Erode", "Deskew", "Custom"])
    
    rate = slider.slider(enhance_type,0.2,8.0,(1.5))

    if enhance_type == 'Original':
        output_image = crop

    elif enhance_type == 'Gray-Scale':
        output_image = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        # output_image = get_grayscale(crop)

    elif enhance_type == 'Contrast':
        enhancer = ImageEnhance.Contrast(crop)
        output_image = enhancer.enhance(rate)

    elif enhance_type == 'Brightness':
        enhancer = ImageEnhance.Brightness(crop)
        output_image = enhancer.enhance(rate)

    elif enhance_type == 'Blurring':
        img = cv2.cvtColor(rgb,1)
        output_image = cv2.GaussianBlur(img,(11,11),rate)
    
    elif enhance_type == 'Cannize':
        output_image = cannize_image(crop)

    elif enhance_type == "Remove_noise":
        output_image = remove_noise(rgb)

    elif enhance_type == "Thresholding":
        output_image = thresholding(gray)

    elif enhance_type == "Dilate":
        output_image = dilate(rgb)

    elif enhance_type == "Opening":
        output_image = opening(rgb)

    elif enhance_type == "Erode":
        output_image = erode(rgb)

    elif enhance_type == "Deskew":
        output_image = deskew(np.array(gray))

    elif enhance_type == "Custom":
        # resized = cv2.resize(gray, interpolation=cv2.INTER_CUBIC)
        dn_gray = cv2.fastNlMeansDenoising(gray, templateWindowSize=7, h=25)
        gray_bin = cv2.threshold(dn_gray, 80, 255, cv2.THRESH_BINARY)[1]
        output_image = gray_bin

    st.image(output_image, width = crop_size, caption = enhance_type)



