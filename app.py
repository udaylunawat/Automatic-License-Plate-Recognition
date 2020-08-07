# Code adapted from https://github.com/fizyr/keras-retinanet
# udaylunawat@gmail.com
# 2020

import streamlit as st
import cv2
import numpy as np
import os
import time
import pytesseract
from PIL import Image,ImageEnhance

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# import miscellaneous modules
from keras import backend as K
from pyngrok import ngrok
import webbrowser
#================================= Functions =================================

# load label to names mapping for visualization purposes
labels_to_names = {0: 'number_plate'}


def e_OCR(crop):
	import easyocr
	reader = easyocr.Reader(['en'])
	ocr_output = reader.readtext(np.array(crop))
	plate_text = ''
	for text in ocr_output:
		plate_text += text[1]
	return plate_text


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

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
	session = K.get_session()
	return model, session

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

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def inference(model, image, scale, session):
	# Run the inference
	start = time.time()

	# set the modified tf session as backend in keras
	# K.set_session(session)
	with session.as_default():
		with session.graph.as_default():
			boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
	st.success("Processing time: {0:.3f} seconds!!".format(time.time() - start))
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
	boxes, scores, labels = inference(model, image, scale, session)
	b, draw = draw_detections(draw, boxes, scores, labels)

	#Write out image
	drawn = Image.fromarray(draw)
	try:
		crop = cv2.rectangle(np.array(draw), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
		crop = crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
		crop = Image.fromarray(crop)
	except:
		drawn = None
		crop = None
	# draw.save(image_output_path)
	# print("Model saved at", image_output_path)

	#   text = reader.readtext(crop)
	#   plate_text = text[0][1]
	return drawn, crop, # plate_text


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

model, session = load_detector_model()
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Indian License Plate detection and recognition ")
st.write("**Using Google's Tensorflow, Retinanet, Streamlit **")

activities = ["Detection", "About"]
choice = st.sidebar.selectbox("Select Task", activities)

if choice == "Detection":
	
	# You can specify more file types below if you want
	img_file_buffer = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)
	st.text("""""")
	try:
		image = Image.open(img_file_buffer)
		# img_array = np.array(image)
		st.write("""
			Preview 👀 Of Selected Image!
			""")
		if image is not None:
			st.image(
				image,
				width = 500,
				caption = 'Original Image'
			)
	except:
		st.warning("Kindly Upload an image")

	metric = st.sidebar.radio("metric ",["Confidence cutoff"])

	# Detections below this confidence will be ignored
	confidence_cutoff = st.sidebar.slider("Cutoff",0.0,1.0,(0.5)) 
	crop = None
	st.text("""""")
	# if st.button("Process"):

	if img_file_buffer is not None:
		try:
			
			result_img, crop = detector(img_file_buffer)
			st.image(
				result_img, 
				caption = 'Annotated with confidence score',
				width = 500
				)
			st.text("""""")
			

		except:
			st.error('''
			Model is not confident enough!
			\nLower confidence cutoff score from sidebar OR Use any other image.
			''')


	if crop is not None:
		st.markdown("<h2 style='text-align: center; color: black;'>\
					Cropped License Plate</h2>", \
						unsafe_allow_html=True)

		enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring","Cannize"])
		st.info(enhance_type)
		if enhance_type == 'Gray-Scale':
			temp = np.array(crop.convert('RGB'))
			# temp = cv2.cvtColor(temp,1)
			output_image = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
			st.image(output_image, width = 500, caption = enhance_type)

		elif enhance_type == 'Contrast':
			c_rate = st.sidebar.slider("Contrast",0.2,8.0,(3.5))
			enhancer = ImageEnhance.Contrast(crop)
			output_image = enhancer.enhance(c_rate)
			st.image(output_image,width = 500, caption = enhance_type)

		elif enhance_type == 'Brightness':
			c_rate = st.sidebar.slider("Brightness",0.2,8.0,(1.5))
			enhancer = ImageEnhance.Brightness(crop)
			output_image = enhancer.enhance(c_rate)
			st.image(output_image, width = 500, caption = enhance_type)

		elif enhance_type == 'Blurring':
			temp = np.array(crop.convert('RGB'))
			blur_rate = st.sidebar.slider("Blur",0.2,8.0,(1.5))
			img = cv2.cvtColor(temp,1)
			output_image = cv2.GaussianBlur(img,(11,11),blur_rate)
			st.image(output_image, width = 500, caption = enhance_type)
		
		elif enhance_type == 'Cannize':
			# cannize = st.sidebar.slider("Cannize",0.2,8.0,(1.5))
			output_image = cannize_image(crop)
			st.image(output_image, width = 500, caption = enhance_type)

		elif enhance_type == 'Original':
			output_image = crop
			st.image(output_image,width = 500, caption = enhance_type)
		
		try:
			tessy_ocr = OCR(output_image)
			if tessy_ocr!='' or tessy_ocr is not None:
				st.write("Google's Tesseract OCR: ", tessy_ocr)
		except:
			pass

		try:
			easy_ocr = e_OCR(output_image)
			st.write("easy OCR: ", easy_ocr)
			st.balloons()
		except:
			pass


		# if text_ocr != '':
		# 	st.success(text_ocr)
			
		

		st.text("""""")
		st.write("Go to the About section from the sidebar to learn more about it.")

elif choice == "About":
	about()
