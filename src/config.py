import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pylab import rcParams
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_random_seed(RANDOM_SEED)



processed_df = pd.read_csv('data/processed/processed.csv')

train_df, test_df = train_test_split(
  processed_df, 
  test_size=0.2, 
  random_state=RANDOM_SEED
)


trainval = [name.split('.')[0] for name in processed_df['image_name']]
train = [name.split('.')[0] for name in train_df['image_name']]
test = [name.split('.')[0] for name in test_df['image_name']]


ANNOTATIONS_FILE = 'data/processed/annotations.csv'
CLASSES_FILE = 'data/processed/classes.csv'
PRETRAINED_MODEL = 'models/snapshots/_pretrained_model.h5'
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'