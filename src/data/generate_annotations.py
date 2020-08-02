from src import config
import urllib

def pretrained_model_load():
    urllib.request.urlretrieve(config.URL_MODEL, config.PRETRAINED_MODEL)
    print('Downloaded pretrained model to ' + config.PRETRAINED_MODEL)


def annot_classes_generator():

    config.train_df.to_csv(config.ANNOTATIONS_FILE, index=False, header=None)
    
    classes = set(['license_plate'])
    with open(config.CLASSES_FILE, 'w') as f:
        for i, line in enumerate(sorted(classes)):
            f.write('{},{}\n'.format(line,i))

def main():
    annot_classes_generator()
    pretrained_model_load()

if __name__ == '__main__':
    
    main()