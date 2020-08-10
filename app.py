"""This is an object detection and ocr app that enables a user to

- select a detection model (in the sidebar),
- upload an image (in the main area)

and get a license plate detection in return.

This app is inspired by the awesome [imageNet](https://github.com/iamatulsingh/imageNet-streamlit)
application developed by [Atul Kumar Singh](https://github.com/iamatulsingh)."""

import streamlit as st
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import time
import pytesseract
from PIL import Image,ImageEnhance
import random
# Machine Learning frameworks
# from keras import backend as K
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# import miscellaneous modules
from pyngrok import ngrok
import webbrowser

# load label to names mapping for visualization purposes
labels_to_names = {0: 'number_plate'}
#================================= Functions =================================


def try_all_OCR(crop_image):
    progress_bar = st.progress(0)
    counter = 0
    for oem in range(0,4):
        for psm in range(0,14):
            counter += 1
            try:
                custom_config = r'--oem {} --psm {}'.format(oem,psm)
                text_output = pytesseract.image_to_string(crop_image, config=custom_config)
                print(custom_config,':',text_output)
                st.write(custom_config,':',text_output)
                progress_bar.progress(counter/(4*14))
            except :
                continue

@st.cache()
def e_OCR(crop):
    import easyocr
    reader = easyocr.Reader(['en'])
    ocr_output = reader.readtext(np.array(crop))
    plate_text = ''
    for text in ocr_output:
        plate_text += text[1]
    return plate_text

@st.cache()
def OCR(crop_image):
    # psm 6 - single line license
    try:
        custom_config = r'--oem 1 --psm 1'
        text_output = pytesseract.image_to_string(crop_image, config=custom_config)
        print(custom_config,':',text_output)
    except:
        pass
    return text_output

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_detector_model():

    model_path = 'output/models/inference/plate_inference.h5'

    # load retinanet model
    print("Loading Model: {}".format(model_path))
    model = models.load_model(model_path, backbone_name='resnet50')

    #Check that it's been converted to an inference model
    try:
        model = models.convert_model(model)
    except:
        print("Model is likely already an inference model")
    # model._make_predict_function()
    # session = K.get_session()
    return model # session

def image_preprocessing(image):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Image formatting specific to Retinanet
    image = preprocess_image(image)
    image, scale = resize_image(image)
    return image, draw, scale
    
def load_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    image = image[:, :, ::-1].copy()
    return image

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def inference(model, image, scale): # session
    # Run the inference
    start = time.time()

    # set the modified tf session as backend in keras
    # K.set_session(session)
    # with session.as_default():
    #     with session.graph.as_default():

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    processing_time = time.time() - start
    st.error("Processing time: {0:.3f} seconds!!".format(processing_time))
    # correct for image scale
    boxes /= scale
    return boxes, scores, labels

def draw_detections(draw, boxes, scores, labels):

    b = None
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < confidence_cutoff:
            break

        #Add boxes and captions
        color = (255, 255, 255)
        thickness = 2
        b = np.array(box).astype(int)

        try:
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

            if(label > len(labels_to_names)):
                st.write("WARNING: Got unknown label, using 'detection' instead")
                caption = "Detection {:.3f}".format(score)
            else:
                caption = "{} {:.3f}".format(labels_to_names[label], score)

            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        
        except TypeError as e:
            st.write("No plate detected")

    return b, draw

def detector(image_path):

    image = load_image(image_path)
    
    image, draw, scale = image_preprocessing(image)
    boxes, scores, labels = inference(model, image, scale) # session
    b, draw = draw_detections(draw, boxes, scores, labels)

    #Write out image
    drawn = Image.fromarray(draw)

    # draw.save(image_output_path)
    # print("Model saved at", image_output_path)

    return drawn, max(scores[0]), draw, b


def cropped_image(image, b):
    crop = cv2.rectangle(np.array(draw), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    crop = crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    crop = Image.fromarray(crop)

    return crop

#============================ About ==========================
def about():
    st.write("""
    ## \u26C5 Behind The Scene
        """)
    st.write("""
    To see how it works, please click the button below!
        """)
    st.text("""""")
    github = st.button("👉🏼 Click Here To See How It Works")
    if github:
        github_link = "https://github.com/udaylunawat/Automatic-License-Plate-Recognition"
        try:
            webbrowser.open(github_link)
        except:
            st.write("""
                ⭕ Something Went Wrong!!! Please Try Again Later!!!
                """)
    st.markdown("Built with Streamlit by [Uday Lunawat](https://github.com/udaylunawat)")

#======================== Time To See The Magic ===========================

crop, image = None, None
img_size, crop_size = 600, 400

st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown("<h1 style='text-align: center; color: black;'>Indian ALPR System using Deep Learning</h1>", unsafe_allow_html=True)
st.info(__doc__)
# st.markdown("<h1 style='text-align: center; color: black;'>Automatic License Plate Recognition</h1>", unsafe_allow_html=True)
# st.markdown("<h2 style='text-align: center; color: black;'>Made with Google's Tensorflow</h2>", unsafe_allow_html=True)
st.write("""""")

st.markdown("👈 Please Open sidebar to upload images")
activities = ["Detection and OCR", "About"]
choice = st.sidebar.selectbox("Select Task", activities)


samplefiles = sorted([sample for sample in listdir('data/sample_images')])
radio = st.sidebar.radio("Choose existing sample or try your own:",('Choose existing', 'Upload'))
if radio == 'Choose existing':
    imageselect = st.sidebar.selectbox("Pick from existing samples", (samplefiles))
    image = Image.open('data/sample_images/'+imageselect)
    IMAGE_PATH = 'data/sample_images/'+imageselect
    image = Image.open('data/sample_images/'+imageselect)
    img_file_buffer = None

else:
    # You can specify more file types below if you want
    img_file_buffer = st.sidebar.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)
    st.text("""""")
    IMAGE_PATH = img_file_buffer
    try:
        image = Image.open(img_file_buffer)
    except:
        pass

    if image == None:
        st.sidebar.success("Upload Image!")
    imageselect = None

if choice == "Detection and OCR" and image:
    
    st.sidebar.text("""
        Preview 👀 Of Selected Image!
        """)

    if image is not None:
        st.sidebar.image(
            image,
            width = 250,
            caption = 'Original Image'
        )

        metric = st.sidebar.radio("metric ",["Confidence cutoff"])

        # Detections below this confidence will be ignored
        confidence_cutoff = st.sidebar.slider("Cutoff",0.0,1.0,(0.5))

    
    st.text("""""")

    # if st.button("Process"):
    model = load_detector_model() # session
    
    if image:
        try:
            annotated_image, score, draw, b = detector(IMAGE_PATH)
            st.subheader("License Plate Detection!")
            st.image(
                annotated_image, 
                caption = 'Annotated Image with confidence score: {0:.2f}'.format(score),
                width = img_size)
            crop = cropped_image(draw, b)
        except:
            st.error('''
            Model is not confident enough!
            \nTry lowering the confidence cutoff score from sidebar OR Use any other image.
            ''')
        
        # st.text()


        if crop is not None:
            st.subheader("Cropped License Plate")

            enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring","Cannize"])


            if enhance_type == 'Original':
                output_image = crop
                st.image(output_image,width = crop_size, caption = enhance_type)

            elif enhance_type == 'Gray-Scale':
                temp = np.array(crop.convert('RGB'))
                # temp = cv2.cvtColor(temp,1)
                output_image = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
                st.image(output_image, width = crop_size, caption = enhance_type)

            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast",0.2,8.0,(3.5))
                enhancer = ImageEnhance.Contrast(crop)
                output_image = enhancer.enhance(c_rate)
                st.image(output_image,width = crop_size, caption = enhance_type)

            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness",0.2,8.0,(1.5))
                enhancer = ImageEnhance.Brightness(crop)
                output_image = enhancer.enhance(c_rate)
                st.image(output_image, width = crop_size, caption = enhance_type)

            elif enhance_type == 'Blurring':
                temp = np.array(crop.convert('RGB'))
                blur_rate = st.sidebar.slider("Blur",0.2,8.0,(1.5))
                img = cv2.cvtColor(temp,1)
                output_image = cv2.GaussianBlur(img,(11,11),blur_rate)
                st.image(output_image, width = crop_size, caption = enhance_type)
            
            elif enhance_type == 'Cannize':
                # cannize = st.sidebar.slider("Cannize",0.2,8.0,(1.5))
                output_image = cannize_image(crop)
                st.image(output_image, width = crop_size, caption = enhance_type)

            st.text("""""")
            st.subheader("Optical Character Recognition (OCR)")
            st.markdown("Note: Here, OCR is performed on the enhanced cropped images.")
            st.text("""""")
            OCR_type = st.sidebar.radio("OCR Mode",["Google's Tesseract OCR","easy_OCR","Secret Combo All-out Attack!!"])
            if st.button('Recognize Characters!!'):
                st.text("""""")

                if OCR_type == "Google's Tesseract OCR":
                    try:
                        tessy_ocr = OCR(output_image)
                        if tessy_ocr!='' and tessy_ocr is not None:
                            st.write("Google's Tesseract OCR: ", tessy_ocr)
                        else:
                            st.write("Google's Tesseract OCR Failed! :sob:")
                    except:
                        pass

                elif OCR_type == "easy_OCR":	
                    try:
                        easy_ocr = e_OCR(output_image)
                        st.write("easy OCR: ", easy_ocr)
                        st.balloons()
                    except:
                        st.write("Easy OCR Failed or not installed! :sob:")

                elif OCR_type == "Secret Combo All-out Attack!!":
                    st.text("""""")
                    try_all_OCR(output_image)
                # if text_ocr != '':
                # 	st.success(text_ocr)
            
        
        
        st.text("""""")
        st.subheader("Go to the About section from the sidebar to learn more about this project!")

elif choice == "About":
    about()
