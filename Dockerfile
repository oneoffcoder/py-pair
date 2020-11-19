FROM continuumio/anaconda3
LABEL author="Jee Vang, Ph.D."
LABEL email="vangjee@gmail.com"
ARG AAPI_VERSION
ARG APYPI_REPO
ENV API_VERSION=$AAPI_VERSION
ENV PYPI_REPO=$APYPI_REPO
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install build-essential -y
COPY . /py-pair
RUN pip install -r /py-pair/requirements.txt
RUN /py-pair/publish.sh