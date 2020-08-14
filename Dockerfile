FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 8080
WORKDIR /usr/app/
RUN make data
RUN make inference_download
RUN make retinanet_source
CMD streamlit run --server.port 8080 --server..enableCORS false app.py 