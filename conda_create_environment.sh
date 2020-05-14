#!/bin/bash
# conda create -y -n license_plate python=3.6.9
# source activate license_plate
# pip install jupyterlab wget pandas Pillow pytesseract numpy opencv-python matplotlib h5py scipy Keras tensorflow tensorflow_gpu
# apt-get install tesseract-ocr
# apt-get install libtesseract-dev

# colab
pip install wget opencv-python matplotlib pytesseract Pillow

apt-get install tesseract-ocr >/dev/null 2>&1
apt-get install libtesseract-dev >/dev/null 2>&1
apt-get install tesseract-ocr-all >/dev/null 2>&1