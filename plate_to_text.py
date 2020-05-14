import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import sys
import time
from PIL import Image
from tensorflow.keras.models import load_model

import pytesseract
# from google.colab.patches import cv2_imshow
# remember to replace cv2_imshow with cv2.imshow when not using colab

from train_character import segment_characters

def show_img(index):
  '''Used to show Images after passing dataframe index'''

  image = cv2.imread("Indian Number Plates/" + df["image_name"].iloc[index])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

  tx = int(df["top_x"].iloc[index] * WIDTH)
  ty = int(df["top_y"].iloc[index] * HEIGHT)
  bx = int(df["bottom_x"].iloc[index] * WIDTH)
  by = int(df["bottom_y"].iloc[index] * HEIGHT)

  image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
  plt.imshow(image)
  plt.show()

def crop_img(image_path):
  # https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

  img = cv2.resize(cv2.imread(image_path) / 255.0, dsize=(WIDTH, HEIGHT))
  # img = cv2.resize(img / 255.0, dsize=(WIDTH, HEIGHT))
  y_hat = model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * WIDTH
  xt, yt = y_hat[0], y_hat[1]
  xb, yb = y_hat[2], y_hat[3]

  img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
  image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 1)
  crop_img = img[int(yt):int(yb), int(xt):int(xb)]
  # plt.imshow(image)
  # plt.imshow(crop_img)
  return (crop_img * 255).astype(np.uint8)


  # get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

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

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


# psm 6 - single line license 
def tessy_ocr(crop_image):
  custom_config = r'--oem 3 --psm 6'
  return pytesseract.image_to_string(crop_image, config=custom_config)

# Adding custom options
def all_combos(preprocess):
  for i in range(14):
    for j in range(4):
      try:
        custom_config = r'--oem '+str(j)+' --psm '+str(i)
        print("\nnumber:{} {}".format(j,i), pytesseract.image_to_string(grey, config=custom_config))
      except:
        continue

def seg_result(seg_char):
  if len(seg_char) == 0:
    return False
  elif len(seg_char<9):
    print("partial success - segmentation")
    return show_results(seg_char)
  return show_results(seg_char)


def output_text():
    if seg_result(seg_char)!=False:
        return seg_result(seg_char)
    else:
        return tessy_ocr(crop_image)

def final(image_path):  
  # image_path = '/content/e.jpg'
  # img = cv2.imread(image_path)
  loaded_image = crop_img(image_path)
  # loaded_image = crop_img(img)
  grey = get_grayscale(loaded_image)
  # canny = canny(grey)
  noise_removed = remove_noise(grey)
  seg_char = segment_characters(loaded_image)
  # text_model = load_model('/content/text_99.h5')
  crop_image = crop_img(image_path)
  text_output = tessy_ocr(crop_image)
  return text_output

if __name__=="__main__":
  
  model = load_model('model.hdf5')
  WIDTH = 224
  HEIGHT = 224
  CHANNEL = 3
  print(f"No. of images passed: {len(sys.argv)-1}")
  for index, image_name in enumerate(sys.argv[1:]):
    license_text = final(image_name)
    print("License text for image {} is {} ".format(image_name, license_text))