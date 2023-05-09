#!/bin/bash

# docker build -f Dockerfile.dlib -t temporal .

PARENTDIR="$(dirname "$PWD")"

nvidia-docker run -it \
    --env LICENSE=yes \
    -p 5555:5555 \
    -v $PWD:/home/jupyter/app \
    -v $PARENTDIR/data:/home/jupyter/data \
    temporal

sudo chown -R $USER:$USER $PARENTDIR/data
sudo chown -R $USER:$USER $PWD
