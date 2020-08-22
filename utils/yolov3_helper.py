import streamlit as st
import cv2
import numpy as np
import config
import time
import pandas as pd

from utils.enhance import cropped_image
from config import NMS_THRESH, LABELS

crop, image = None, None

# Initialization
# load the COCO class labels our YOLO model was trained on

# derive the paths to the YOLO weights and model configuration
weightsPath = config.MODEL_PATH
configPath = config.CONFIG_PATH

def yolo_detector(frame, net, ln, MIN_CONF, Idx=0):
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
    st.sidebar.success("Processing time for YOLOV3: {} {:.4f} seconds.".format('\n',time.time() - start))
    print("Processing time for YOLOV3: --- {:.4f} seconds ---".format(time.time() - start))
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

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_network(configpath, weightspath):

    with st.spinner("Loading Yolo weights!"):
        # load our YOLO object detector trained on our dataset (1 class)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # determine only the *output* layer names that we need from YOLO
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layer_names

def yolo_crop_correction(frame, bbox, w, h):
    # resizing cropped image
    (startX, startY, endX, endY) = bbox
    
    crop = cropped_image(frame, (startX, startY, endX, endY))

    crop_w, crop_h = endX - startX, endY - startY # height & width of number plate of 416*416 image
    width_m, height_m = w/416, h/416 # width and height multiplier
    w2, h2 = round(crop_w*width_m), round(crop_h*height_m)
    crop = cv2.resize(np.asarray(crop), (w2, h2))
    return crop
    
def yolo_inference(image, confidence_cutoff):
    # YOLO Detection
    # Preprocess
    frame = cv2.resize(np.asarray(image), (416, 416))

    # Get parameter
    MIN_CONF = confidence_cutoff
    w, h = image.size

    net, output_layer_names = load_network(configPath, weightsPath)
    results = yolo_detector(frame, net, output_layer_names, MIN_CONF, Idx=LABELS.index("number_plate"))

    crop_list = []
    # Loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # Extract the bounding box and centroid coordinates
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid

        # crop correct and multiple image cropping
        try:
            crop = yolo_crop_correction(frame, bbox, w, h)
            crop_list.append([crop, prob])
            
        except NameError as e:
            st.error('''
            Model is not confident enough!
            \nTry lowering the confidence cutoff score from sidebar.
            ''')
            st.error("Error log: "+str(e))

        # Overlay
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
        # cv2.circle(frame, (cX, cY), 5, (0, 255, 0), 1)

    # Show result
    image = cv2.resize(np.asarray(frame), (w, h)) # resizing image as yolov3 gives 416*416 as output
    
    return image, crop_list