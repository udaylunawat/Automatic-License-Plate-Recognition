FROM conda/miniconda3
RUN apt-get update  -y
RUN apt-get upgrade -y
RUN apt-get -y install wget

# setting work directory and copying content
WORKDIR /app
ADD requirements.txt /app/requirements.txt
ADD . /app

# Packages for make
RUN apt-get update && \
    apt-get -y install build-essential

# updating pip and installing git
RUN pip install --upgrade pip
RUN apt -y install git

# Generating data ETL, downloading inference and installing retinanet from source
RUN make data
RUN make inference_download
RUN make retinanet_source

EXPOSE 8080

CMD streamlit run --server.port 8080 --server..enableCORS false app.py 