FROM continuumio/miniconda3
# LABEL maintainer="Uday Lunawat @dracarys3"

# Packages for make
RUN sudo apt-get update -y && sudo apt-get upgrade -y && sudo apt-get -y install wget git build-essential libsm6 libxext6 libxrender-dev libgl1-mesa-glx software-properties-common && sudo add-apt-repository -y ppa:alex-p/tesseract-ocr && sudo apt-get -y install p7zip-full tesseract-ocr libtesseract-dev 

# setting work directory and copying content
WORKDIR /app
ADD requirements.txt /app/requirements.txt
ADD . /app

# Generating data ETL, downloading inference and installing retinanet from source
RUN make -s ETL

EXPOSE 8080

CMD streamlit run --server.port 8888 --server.enableCORS false app.py 