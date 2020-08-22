import streamlit as st
import cv2
from PIL import Image
import time
import numpy as np

# Machine Learning frameworks
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from config import labels_to_names
from utils.enhance import cropped_image

crop, image = None, None

#================================= Functions =================================

@st.cache(suppress_st_warning=True, allow_output_mutation=False, show_spinner=False)
def load_retinanet():
    # caching.clear_cache()
    model_path = 'output/models/inference/plate_inference_tf2.h5'

    # load retinanet model
    print("Loading Model: {}".format(model_path))
    with st.spinner("Loading retinanet weights!"):
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
    st.sidebar.success("Processing time for RetinaNet: {} {:.4f} seconds.".format("\n",time.time() - start))
    print("Processing time for RetinaNet: {:.4f} seconds ".format(time.time() - start))

    # correct for image scale
    boxes /= scale
    return boxes, scores, labels


def draw_detections(draw, boxes, scores, labels, confidence_cutoff):
    draw2 = draw.copy()
    crop_list = []
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
            crop_list.append([cropped_image(draw2, (b[0], b[1], b[2], b[3]) ), score])

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
        
    return (b[0], b[1], b[2], b[3]), draw, crop_list


def retinanet_detector(image_path, model, confidence_cutoff):

    image = load_image(image_path)
    
    image, draw, scale = image_preprocessing(image)
    boxes, scores, labels = inference(model, image, scale) # session
    b, draw, crop_list = draw_detections(draw, boxes, scores, labels, confidence_cutoff)

    #Write out image
    drawn = Image.fromarray(draw)

    # draw.save(image_output_path)
    # print("Model saved at", image_output_path)

    return drawn, max(scores[0]), crop_list