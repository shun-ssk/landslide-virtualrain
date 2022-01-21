FROM python:3.8.8
USER root

RUN apt-get install -y vim less
# 解析に必要なパッケージを記述
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip -m install numpy && \
    pip -m install matplotlib