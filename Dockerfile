FROM alpine
# LABEL maintainer="Uday Lunawat @dracarys3"

# Packages for make
RUN apt-get update -y && apt-get upgrade -y && apt-get -y install wget git build-essential libsm6 libxext6 libxrender-dev libgl1-mesa-glx software-properties-common && add-apt-repository -y ppa:alex-p/tesseract-ocr && apt-get -y install p7zip-full tesseract-ocr libtesseract-dev 

# setting work directory and copying content
WORKDIR /app
ADD requirements.txt /app/requirements.txt
ADD . /app

# Generating data ETL, downloading inference and installing retinanet from source
RUN make -s ETL

EXPOSE 8080

CMD streamlit run --server.port 8888 --server.enableCORS false app.py 