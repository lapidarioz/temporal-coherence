#!/bin/bash

# docker build -t temporal .

nvidia-docker run -u $(id -u):$(id -g) -it \
    --env LICENSE=yes \
    -p 8888:8888 \
    -v $PWD:/home/jupyter/app \
    -v /home/rafa/data:/home/jupyter/data \
    temporal
