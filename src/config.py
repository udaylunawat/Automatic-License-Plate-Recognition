import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42

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
PRETRAINED_MODEL = 'model/snapshots/_pretrained_model.h5'
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'