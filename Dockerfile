FROM dracarys3/alpr-india:3.2-tf2.2.0-jupyter
LABEL maintainer="Uday Lunawat @dracarys3"

RUN apt-get update -y && apt-get upgrade -y

# setting work directory and copying content
WORKDIR /app
ADD . /app

# Generating data ETL, downloading inference and installing retinanet from source
RUN make -s ETL

EXPOSE 8080

CMD streamlit run serve/app.py --server.port 8080 --server.enableXsrfProtection=false --server.enableCORS=false