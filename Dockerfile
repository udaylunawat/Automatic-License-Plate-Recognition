FROM continuumio/miniconda3
# LABEL maintainer="Uday Lunawat @dracarys3"
RUN apt-get update  -y && apt-get upgrade -y

# Packages for make
RUN apt-get -y install --upgrade pip wget git build-essential \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    software-properties-common
RUN add-apt-repository ppa:alex-p/tesseract-ocr

# setting work directory and copying content
WORKDIR /app
ADD requirements.txt /app/requirements.txt
ADD . /app

# 7zip and tesseract
RUN	apt-get -y install p7zip-full tesseract-ocr libtesseract-dev

# Generating data ETL, downloading inference and installing retinanet from source
RUN make -s data
RUN make -s inference_download
RUN make -s retinanet_source

EXPOSE 8080

CMD streamlit run --server.port 8888 --server.enableCORS false app.py 