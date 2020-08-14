FROM conda/miniconda3
RUN apt-get update  -y
RUN apt-get upgrade -y
RUN apt-get -y install wget
# Packages for make
RUN apt-get update && \
    apt-get -y install build-essential \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    software-properties-common
RUN add-apt-repository ppa:alex-p/tesseract-ocr

# setting work directory and copying content
WORKDIR /app
ADD requirements.txt /app/requirements.txt
ADD . /app




RUN	apt-get -y install p7zip-full
RUN	apt-get -y install tesseract-ocr
RUN	apt-get -y install libtesseract-dev

# updating pip and installing git
RUN pip install --upgrade pip
RUN apt -y install git

# Generating data ETL, downloading inference and installing retinanet from source
RUN make data
RUN make inference_download
RUN make retinanet_source

EXPOSE 8080

CMD streamlit run --server.port 8888 --server.enableCORS false app.py 