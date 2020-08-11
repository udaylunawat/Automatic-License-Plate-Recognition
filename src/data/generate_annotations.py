from src.config import PRETRAINED_MODEL, URL_MODEL
import urllib

def pretrained_model_load():
    if os.path.exists(PRETRAINED_MODEL):
        print('Model already downloaded, skipping download.')
    else:
        urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)
        print('Downloaded pretrained model to ' + PRETRAINED_MODEL)


def annot_classes_generator():
    train_df = config.train_df.copy()
    train_df.image_name = train_df.image_name.apply(lambda x: 'VOC/JPEGImages/'+x)
    train_df.to_csv(config.ANNOTATIONS_FILE, index=False, header=False)
    
    classes = set(['license_plate'])
    with open(config.CLASSES_FILE, 'w') as f:
        for i, line in enumerate(sorted(classes)):
            f.write('{},{}\n'.format(line,i))

def main():
    annot_classes_generator()
    pretrained_model_load()

if __name__ == '__main__':
    
    main()