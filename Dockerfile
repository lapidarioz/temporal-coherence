FROM tensorflow/tensorflow:latest-gpu

# Adapted from https://github.com/d1egoprog/docker-tensorflow-gpu-jupyter/blob/main/tensorflow-gpu-jupyter/Dockerfile

RUN apt-get update && apt-get install -y \
        ffmpeg \
        python3-tk \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev

USER root

RUN adduser jupyter

USER jupyter

ENV PORT 8888

ENV PATH="${PATH}:/home/jupyter/.local/bin"

WORKDIR /home/jupyter

RUN python3 -m pip install --no-cache --upgrade setuptools pip

RUN pip install jupyter && pip install jupyterlab

COPY requirements.txt /home/jupyter/


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
