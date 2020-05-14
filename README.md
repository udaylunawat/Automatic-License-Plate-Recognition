# Automatic-License-Plate-Recognition

The project developed using TensorFlow to detect the License Plate from a car and uses the Tesseract Engine to recognize the charactes from the detected plate.

Software Packages Needed
- Anaconda 3 (Tool comes with most of the required python packages along with python3)
- Tesseract Engine (Must need to be installed)

Python Packages Needed
- Tensorflow
- openCV
- pytesseract

## Instructions

- Run these scripts to clone the repository, install dependencies and train the detectors.

```
!git clone https://github.com/udaylunawat/License_plate_detector_Deep_Learning.git
!bash conda_create_environment.sh
!bash train.sh
```
- Run this script and pass image as cli to get result

```
python plate_to_text.py car.jpg
```



