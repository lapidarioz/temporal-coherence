#!/bin/bash

# docker build -t temporal .

nvidia-docker run -it \
    --env LICENSE=yes \
    -p 5555:5555 \
    -v $PWD:/home/jupyter/app \
    -v /home/rafa/data:/home/jupyter/data \
    temporal
