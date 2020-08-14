FROM conda/miniconda3
RUN apt-get update  -y
RUN apt-get upgrade -y
RUN apt-get -y install wget

ARG JSON_DOWNLOAD_URL=https://www.dropbox.com/s/8netfite5znq6o4/Indian_Number_plates.json
ARG IMAGES_ZIP=https://www.dropbox.com/s/k3mhm1kz192bwue/Indian_Number_Plates.7z
ARG INFERENCE=https://storage.googleapis.com/dracarys3_bucket/ALPR/license_plate/ALPR/retinanet_inference/plate_inference_tf2_2.h5

# RUN useradd -ms /bin/bash admin
WORKDIR /app
COPY requirements.txt ./
COPY . ./

RUN apt-get update && \
    apt-get -y install build-essential

RUN pip install --upgrade pip
RUN pip install --ignore-installed -r requirements.txt


EXPOSE 8080


# RUN make data
# RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository universe

RUN apt-get -y install p7zip-full
RUN apt-get -y install tesseract-ocr
RUN apt-get -y install libtesseract-dev
RUN pip install pip setuptools wheel --progress-bar off

# RUN pip install -r requirements.txt --progress-bar off | grep -v 'already satisfied'


# RUN chown -R admin:admin /app
# RUN chmod 755 /app
# USER admin


RUN	mkdir -p data/0_raw data/1_external data/2_interim data/3_processed data/0_raw/Indian_Number_Plates
RUN	mkdir -p data/3_processed/VOC/Annotations data/3_processed/VOC/JPEGImages data/3_processed/VOC/ImageSets/Main
RUN	mkdir -p output/models/TrainingOutput output/models/snapshots output/models/inference


RUN	wget -c ${JSON_DOWNLOAD_URL} -O data/0_raw/Indian_Number_plates.json -q --show-progress
RUN	wget -c ${IMAGES_ZIP} -P data/0_raw -q --show-progress
RUN	7z x -y data/0_raw/Indian_Number_Plates.7z -odata/0_raw/Indian_Number_Plates

# RUN chmod +X src/data/make_dataset.py
# RUN chmod -R u+rwX,go+rwX /app
RUN python3 src/data/make_dataset.py


RUN python3 src/data/preprocess.py -i data/0_raw/ -o data/3_processed/
RUN python3 src/data/generate_annotations.py


RUN	cp data/0_raw/Indian_Number_Plates/* data/3_processed/VOC/JPEGImages
RUN python3 src/data/generate_pascalvoc.py data/0_raw/Indian_Number_plates.json data/3_processed/VOC/JPEGImages data/3_processed/VOC/Annotations

# RUN make inference_download
RUN	wget -c ${INFERENCE} -O output/models/inference/plate_inference_tf2.h5 -q --show-progress

# RUN make retinanet_source
RUN apt -y install git
RUN	git clone https://github.com/udaylunawat/keras-retinanet.git
RUN	pip install keras-retinanet/.
RUN	cd keras-retinanet
RUN	python setup.py build_ext --inplace
RUN	cd ..



CMD streamlit run --server.port 8080 --server..enableCORS false app.py 