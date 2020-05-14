#!/bin/bash
#conda create -y -n license_plate python=3.6.9
source activate license_plate
conda install -y jupyterlab wget pandas numpy opencv-python matplotlib h5py scipy Keras==2.3.1 tensorflow==2.2.0 tensorflow_gpu
pip install pytesseract
pip install Pillow
!apt install tesseract-ocr
!apt install libtesseract-dev