"""This is an Object detection and Optical Character Recognition(OCR) app that enables a user to

- Select or upload an image.
- Get a annotated and cropped license plate image.
- Play around with Image enhance options (OpenCV).
- Get OCR Prediction using various options.

"""

import streamlit as st
st.beta_set_page_config(page_title="Ex-stream-ly Cool App", page_icon="😎",layout="centered",initial_sidebar_state="expanded")
import os
from os import listdir
from os.path import isfile, join

import config
import cv2
import numpy as np

import time
import pytesseract
from PIL import Image,ImageEnhance
import random
import pandas as pd
# Machine Learning frameworks
from tensorflow.keras import backend as K
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# import miscellaneous modules
from pyngrok import ngrok
import webbrowser

# https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True

# load label to names mapping for visualization purposes
labels_to_names = {0: 'number_plate'}


# YOLO Constant
MIN_CONF = 0.5
NMS_THRESH = 0.3
#================================= Functions =================================

@st.cache()
def try_all_OCR(crop_image):
    progress_bar = st.progress(0)
    counter = 0
    for oem in range(0,4):
        for psm in range(0,14):
            counter += 1
            try:
                custom_config = r'--oem {} --psm {}'.format(oem,psm)
                text_output = pytesseract.image_to_string(crop_image, config=custom_config)
                st.warning(custom_config+':'+text_output)
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

def streamlit_OCR(output_image):
    st.text("""""")
    st.write("## 🎅 Bonus:- Optical Character Recognition (OCR)")
    st.text("""""")
    
    OCR_type = st.sidebar.radio("OCR Mode",["easy_OCR","Google's Tesseract OCR","Secret Combo All-out Attack!!"])
    st.warning("Note: OCR is performed on the enhanced cropped images.")

    if OCR_type == "Google's Tesseract OCR":
        st.error("Researching Google's Tesseract OCR is a work in progress 🚧\
                \nThe results might be unreliable.")
    st.text("""""")
    placeholder = st.empty()
    note = st.empty()
    st.text("""""")
    if st.button('Recognize Characters !!'):
        
        if OCR_type == "Google's Tesseract OCR":
            # try:
            tessy_ocr = OCR(output_image)
            
            if len(tessy_ocr) == 0:
                placeholder.error("Google's Tesseract OCR Failed! :sob:")
            else:
                placeholder.success("Google's Tesseract OCR: " + tessy_ocr)
                st.text("""""")


        elif OCR_type == "easy_OCR":	
            try:
                easy_ocr = e_OCR(output_image)
                
                st.text("""""")
                placeholder.success("easy OCR: " + easy_ocr)
                st.balloons()
            
            except ModuleNotFoundError:
                placeholder.error("EasyOCR not installed")
            except:
                placeholder.error("Easy OCR Failed! :sob:")

        elif OCR_type == "Secret Combo All-out Attack!!":
            st.text("""""")
            try_all_OCR(output_image)
    

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def load_detector_model():

    model_path = 'output/models/inference/plate_inference_tf2.h5'

    # load retinanet model
    print("Loading Model: {}".format(model_path))
    model = models.load_model(model_path, backbone_name='resnet50')

    #Check that it's been converted to an inference model
    try:
        model = models.convert_model(model)
    except:
        print("Model is likely already an inference model")
    return model 

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

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def inference(model, image, scale): # session
    # Run the inference
    
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    st.error("Processing time for RetinaNet: --- {0:.4f} seconds ---".format(time.time() - start))

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
            # startX, startY, endX, endY = b[0], b[1], b[2], b[3]
        except TypeError as e:
            st.error("No plate detected")
        
    return (b[0], b[1], b[2], b[3]), draw


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


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def cropped_image(image, b):
    crop = cv2.rectangle(np.array(image), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    crop = crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    crop = Image.fromarray(crop)

    return crop

def enhance_crop(crop):

    if crop is not None:
        st.subheader("Cropped License Plate")

        enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring","Cannize"])


        if enhance_type == 'Original':
            output_image = crop
            st.image(output_image, width=crop_size, caption = enhance_type)

        elif enhance_type == 'Gray-Scale':
            temp = np.array(crop.convert('RGB'))
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
            output_image = cannize_image(crop)
            st.image(output_image, width = crop_size, caption = enhance_type)
            


def yolo_detect(frame, net, ln, Idx=0):
    # grab the dimensions of the frame and  initialize the list of
    # results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(ln)
    st.error("Processing time for YOLOV3: --- {0:.4f} seconds ---".format(time.time() - start))
    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == Idx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results


def streamlit_preview_image(image):
    placeholder = st.sidebar.empty()
    placeholder.image(
    image,
    use_column_width=True,
    caption = "Original Image")

def streamlit_output_image(image, caption):
    placeholder = st.empty()
    placeholder.image(
    image,
    use_column_width=True,
    caption = caption)

#============================ About ==========================
def about():
    st.success("Deployed this Streamlit app with Docker on GCP (Google Cloud Platform) ")
    st.warning("""
    ## \u26C5 Behind The Scenes
        """)
    st.success("""
    To see how it works, please click the button below!
        """)
    st.text("""""")
    github = st.button("👉🏼 Click Here To See How It Works")
    if github:
        github_link = "https://github.com/udaylunawat/Automatic-License-Plate-Recognition"
        try:
            webbrowser.open(github_link)
        except:
            st.error("""
                ⭕ Something Went Wrong!!! Please Try Again Later!!!
                """)
    st.info("Built with Streamlit by [Uday Lunawat 😎](https://github.com/udaylunawat)")

def about_yolo():
    yolo_dir = 'banners/yolo/'
    yolo_banner = random.choice(listdir(yolo_dir))
    st.sidebar.markdown("\n")
    st.sidebar.image(yolo_dir+yolo_banner, use_column_width=True)
    
    st.sidebar.info(
        "**YOLO (“You Only Look Once”)** is an effective real-time object recognition algorithm, \
        first described in the seminal 2015 [**paper**](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al.\
        It's a network that uses **Deep Learning (DL)** algorithms for **object detection**. \
        \n\n[**YOLO**](https://missinglink.ai/guides/computer-vision/yolo-deep-learning-dont-think-twice/) performs object detection \
        by classifying certain objects within the image and **determining where they are located** on it.\
        \n\nFor example, if you input an image of a herd of sheep into a YOLO network, it will generate an output of a vector of bounding boxes\
            for each individual sheep and classify it as such. Yolo is based on algorithms based on regression━they **scan the whole image** and make predictions to **localize**, \
        identify and classify objects within the image. \
        \n\nAlgorithms in this group are faster and can be used for **real-time** object detection.\
        **YOLO V3** is an **improvement** over previous YOLO detection networks. \
        Compared to prior versions, it features **multi-scale detection**, stronger feature extractor network, and some changes in the loss function.\
        As a result, this network can now **detect many more targets from big to small**. \
        And, of course, just like other **single-shot detectors**, \
        YOLO V3 also runs **quite fast** and makes **real-time inference** possible on **GPU** devices.")

def about_retinanet():
    od_dir = 'banners/Object detection/'
    od_banner = random.choice(listdir(od_dir))
    st.sidebar.markdown("\n")
    st.sidebar.image(od_dir+od_banner, use_column_width=True)

    st.sidebar.info(
        "[RetinaNet](https://arxiv.org/abs/1708.02002) is **one of the best one-stage object detection models** that has proven to work well with dense and small scale objects. \
        For this reason, it has become a **popular** object detection model to be used with aerial and satellite imagery. \
        \n\n[RetinaNet architecture](https://www.mantralabsglobal.com/blog/better-dense-shape-detection-in-live-imagery-with-retinanet/) was published by **Facebook AI Research (FAIR)** and uses Feature Pyramid Network (FPN) with ResNet. \
        This architecture demonstrates **higher accuracy** in situations where *speed is not really important*. \
        RetinaNet is built on top of FPN using ResNet.")

def model_select():
    crop, image = None, None
    st.write("""""")
    st.write("## Upload your own image")
    samplefiles = sorted([sample for sample in listdir('data/sample_images')])
    radio_list = ['Choose existing', 'Upload']

    query_params = st.experimental_get_query_params()
    # Query parameters are returned as a list to support multiselect.
    # Get the second item (upload) in the list if the query parameter exists.
    # Setting default page as Upload page, checkout the url too. The page state can be shared now!
    default = int(query_params['activity'][0]) if 'activity' in query_params else 1

    activity = st.radio("Choose existing sample or try your own:",radio_list,index=default)
    if activity:
        st.experimental_set_query_params(activity=radio_list.index(activity))
        if activity == 'Choose existing':
            imageselect = st.selectbox("Pick from existing samples", (samplefiles))
            image = Image.open('data/sample_images/'+imageselect)
            IMAGE_PATH = 'data/sample_images/'+imageselect
            image = Image.open('data/sample_images/'+imageselect)
            img_file_buffer = None

        else:
            # You can specify more file types below if you want
            img_file_buffer = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)
            st.text("""""")
            IMAGE_PATH = img_file_buffer
            try:
                image = Image.open(img_file_buffer)
            except:
                pass

            if image == None:
                st.success("Upload Image!")
            imageselect = None

    return image, imageselect, IMAGE_PATH

#======================== Time To See The Magic ===========================


crop, image = None, None
img_size, crop_size = 600, 400

st.set_option('deprecation.showfileUploaderEncoding', False)





activities = ["Home", "RetinaNet Detection", "YoloV3 Detection", "About"]
choice = st.sidebar.radio("Go to", activities)




if choice == "Home":
    st.markdown("<h1 style='text-align: center; color: black;'>Indian ALPR System using Deep Learning 👁</h1>", unsafe_allow_html=True)
    st.sidebar.info(__doc__)
    st.write("## How does it work?")
    st.write("Add an image of a car and a [deep learning](http://wiki.fast.ai/index.php/Lesson_1_Notes) model will look at it and find the **license plate** like the example below:")
    st.image(Image.open("output/sample_output.png"),
            caption="Example of a model being run on a car.",
            use_column_width=True)

    st.text("""""")
    st.write("## How is this made?")

    st.text("""""")
    banners = 'banners/'
    files = [banners+f for f in os.listdir(banners) if os.path.isfile(os.path.join(banners, f))]
    st.image(random.choice(files),use_column_width=True)

    st.sidebar.info("- The learning (detection) happens  \
                    with a fine-tuned [**Retinanet**](https://arxiv.org/abs/1708.02002) or a [**YoloV3**](https://pjreddie.com/darknet/yolo/) \
                    model ([**Google's Tensorflow 2**](https://www.tensorflow.org/)), \
                    \n- This front end (what you're reading) is built with [**Streamlit**](https://www.streamlit.io/) \
                    \n- It's all hosted on the cloud using [**Google Cloud Platform's App Engine**](https://cloud.google.com/appengine/).")
    # st.video("https://youtu.be/C_lIenSJb3c")
    #  and a [YouTube playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFeMrZ0sBjmnUBZNX9xaqKuM) detailing more below.")
    # or OpenCV Haar cascade


    st.write("""""")
    st.sidebar.warning("#### Checkout the Source code on [GitHub](https://github.com/udaylunawat/Automatic-License-Plate-Recognition)")

K.clear_session()
st.text("""""")
model = load_detector_model()

if choice == "RetinaNet Detection":
    image, imageselect, IMAGE_PATH = model_select()
    if image is None:
        about_retinanet()

    if image is not None:
        if st.sidebar.checkbox("View Documentation"):
            about_retinanet()

        st.sidebar.markdown("## Preview Of Selected Image! 👀")
        streamlit_preview_image(image)

        metric = st.sidebar.radio("metric ",["Confidence cutoff"])

        # Detections below this confidence will be ignored
        confidence_cutoff = st.sidebar.slider("Cutoff",0.0,1.0,(0.5))

    
    st.text("""""")

    
    st.warning("**Note:** The model has been trained on Indian cars and number plates, and therefore will only work with those kind of images.")
    st.text("""""")
    # if st.button("Make a Prediction 🔥"):
    
    if image:
        try:
            with st.spinner('Doing the Math...'):
                annotated_image, score, draw, b = detector(IMAGE_PATH)
            st.subheader("License Plate Detection!")
            streamlit_output_image(annotated_image, 'Annotated Image with confidence score: {0:.2f}'.format(score))
            crop = cropped_image(draw, b)
        except TypeError:
            st.error('''
            Model is not confident enough!
            \nTry lowering the confidence cutoff score from sidebar.
            ''')


        enhance_crop(crop)
        streamlit_OCR(crop)


if choice == "YoloV3 Detection":
    image, imageselect, IMAGE_PATH = model_select()
    if image is None:
        about_yolo()
    
    if image is not None:
        if st.sidebar.checkbox("View Documentation"):
            about_yolo()
        
        st.sidebar.markdown("## Preview Of Selected Image! 👀")
        streamlit_preview_image(image)

        metric = st.sidebar.radio("metric ",["Confidence cutoff"])

        # Detections below this confidence will be ignored
        confidence_cutoff = st.sidebar.slider("Cutoff",0.0,1.0,(0.5))

    
    st.text("""""")

    
    st.warning("**Note:** The model has been trained on Indian cars and number plates, and therefore will only work with those kind of images.")
    st.text("""""")
    
    

    if image is not None:

        # YOLO Detection
        # Preprocess
        frame = cv2.resize(np.asarray(image), (416, 416))

        # Get parameter
        MIN_CONF = confidence_cutoff
        show_prob = st.sidebar.checkbox('Show Probability')
        w, h = image.size

        # Inference
        # if st.button('Run Inference'):

        # Initialization
        # load the COCO class labels our YOLO model was trained on
        labelsPath = config.LABEL_PATH
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = config.MODEL_PATH
        configPath = config.CONFIG_PATH

        # load our YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        results = yolo_detect(frame, net, ln, Idx=LABELS.index("number_plate"))

        if show_prob:
            # Show detection results in dataframe
            probs = [result[0] for result in results]
            df = pd.DataFrame(dict(ID=list(range(len(results))), Prob=probs))
            st.dataframe(df)

            # Simple plot
            st.line_chart(df['Prob'])

        # Loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # Extract the bounding box and centroid coordinates
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid

            # Overlay
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.circle(frame, (cX, cY), 5, (0, 255, 0), 1)

        # Show result
        image = cv2.resize(np.asarray(frame), (w, h)) # resizing image as yolov3 gives 416*416 as output
        streamlit_output_image(image, "YoloV3 Output")

        try:
            crop = cropped_image(frame, (startX, startY, endX, endY))
            # crop = cv2.resize(np.asarray(crop), (w, h))
        except TypeError:
            st.error('''
            Model is not confident enough!
            \nTry lowering the confidence cutoff score from sidebar.
            ''')
        enhance_crop(np.array(crop))
        streamlit_OCR(crop)

elif choice == "About":
    about()


