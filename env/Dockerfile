FROM python:3.8.8
USER root

RUN apt-get install -y vim less
# 解析に必要なパッケージを記述
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip -m install numpy && \
    pip -m install matplotlib

# 空のディレクトリ作成
RUN mkdir -p /root/src/input/utils && \
    mkdir -p /root/src/output/{ pod/txt, pod/png/mode, makeRain }