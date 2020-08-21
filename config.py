import numpy as np

## streamlit local build
# DIR_PATH = '/content/Automatic-License-Plate-Recognition'

# Docker Path
DIR_PATH = ''

model = 'output/models/inference/yolov3-custom_last.weights'
model_config = 'cfg/yolov3-custom.cfg'
labels = 'cfg/obj.names'
# input_videos = 'videos/'
# output_video = 'output/output_video.mp4'

MODEL_PATH = model
CONFIG_PATH = DIR_PATH + model_config
LABEL_PATH = DIR_PATH + labels
# OUTPUT_PATH = DIR_PATH + output_video
# INPUT_PATH = DIR_PATH+ input_videos
# VIDEO_PATH = DIR_PATH + input_videos


# RETINANET CONSTANTS
##################################
# load label to names mapping for visualization purposes

labels_to_names = {0: 'number_plate'}



# YOLO CONSTANTS
###################################

MIN_CONF = 0.5
NMS_THRESH = 0.3

DEFAULT_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3

LABELS = open(LABEL_PATH).read().strip().split('\n')
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = 'uint8')