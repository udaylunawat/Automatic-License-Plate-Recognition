# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from pyngrok import ngrok
import easyocr
reader = easyocr.Reader(['en'])


# Code adapted from https://github.com/fizyr/keras-retinanet
# jasperebrown@gmail.com
# 2020

# This script loads a single image, runs inferencing on it
# and saves that image back out with detections overlaid.

# You need to set the model_path and image_path below

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# import miscellaneous modules
import cv2
import os
import numpy as np
import time
from PIL import Image

import tensorflow as tf

from keras import backend
backend.clear_session()

# set tf backend to allow memory to grow, instead of claiming everything
# import tensorflow as tf


def detect(image_path):
  
  model_path = '/content/Automatic-License-Plate-Recognition/models/inference/plate_inference.h5'
  confidence_cutoff = 0.5 # Detections below this confidence will be ignored

  def get_session():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      return tf.Session(config=config)

  # use this environment flag to change which GPU to use
  #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  # set the modified tf session as backend in keras
  keras.backend.tensorflow_backend.set_session(get_session())

  # adjust this to point to your downloaded/trained model
  # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
  #model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

  print("Loading image from {}".format(image_path))
  image = np.asarray(Image.open(image_path).convert('RGB'))
  image = image[:, :, ::-1].copy()

  # load retinanet model
  print("Loading Model: {}".format(model_path))
  model = models.load_model(model_path, backbone_name='resnet50')

  #Check that it's been converted to an inference model
  try:
      model = models.convert_model(model)
  except:
      print("Model is likely already an inference model")

  # load label to names mapping for visualization purposes
  labels_to_names = {0: 'number_plate'}

  # copy to draw on
  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  # Image formatting specific to Retinanet
  image = preprocess_image(image)
  image, scale = resize_image(image)

  # Run the inference
  start = time.time()

  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  print("processing time: ", time.time() - start)

  # correct for image scale
  boxes /= scale

  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < confidence_cutoff:
          break

      #Add boxes and captions
      color = (255, 255, 255)
      thickness = 2
      b = np.array(box).astype(int)
      cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

      if(label > len(labels_to_names)):
          st.write("WARNING: Got unknown label, using 'detection' instead")
          caption = "Detection {:.3f}".format(score)
      else:
          caption = "{} {:.3f}".format(labels_to_names[label], score)

      cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
      cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

  #Write out image
  # draw = Image.fromarray(draw)
  drawn = cv2.cvtColor(np.array(draw), cv2.COLOR_BGR2RGB)

  crop = cv2.rectangle(np.array(draw), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
  crop = crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
  # crop = (crop * 255).astype(np.uint8)
  # draw.save(image_output_path)
  # print("Model saved at", image_output_path)

  text = reader.readtext(crop)
  plate_text = text[0][1]
  return drawn, crop, plate_text

def about():
	st.write(
		'''

		''')


def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Indian License Plate detection and recognition :sunglasses:")
    st.write("**Using Google's Tensorflow, Retinanet, Streamlit and ** :love:")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
    			result_img, crop, plate_text = detect(image_file)

    			st.image([result_img, crop], use_column_width = True)
    			st.write("Found plate!!\n", plate_text)
          
    elif choice == "About":
    	about()

if __name__ == "__main__":
    
    main()