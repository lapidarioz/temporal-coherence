#!/bin/bash

# docker build -f Dockerfile.dlib -t temporal .
# docker build -f Dockerfile.dlib -t tensorflow .

PARENTDIR="$(dirname "$PWD")"


nvidia-docker run -it \
    --env LICENSE=yes \
    -p 5555:5555 \
    -v $PWD:/home/jupyter/app \
    -v $PARENTDIR/data:/home/jupyter/data \
    -w /home/jupyter/app \
    tensorflow

# nvidia-docker run -it \
#     --env LICENSE=yes \
#     -p 5555:5555 \
#     -v $PWD:/home/jupyter/app \
#     -v $PARENTDIR/data:/home/jupyter/data \
#     -w /home/jupyter/app \
#     temporal

sudo chown -R $USER:$USER $PARENTDIR/data
sudo chown -R $USER:$USER $PWD
