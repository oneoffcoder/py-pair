FROM continuumio/anaconda3
LABEL Jee Vang, Ph.D. "vangjee@gmail.com"
ARG AAPI_VERSION
ARG APYPI_REPO
ENV API_VERSION=$AAPI_VERSION
ENV PYPI_REPO=$APYPI_REPO
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install build-essential -y
COPY . /py-pair
RUN conda install --file /py-pair/requirements.txt -y
RUN /py-pair/publish.sh