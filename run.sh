#!/bin/bash

# docker build -f Dockerfile.dlib -t temporal .
# docker build -f Dockerfile.tensorflow -t tensorflow .

PARENTDIR="$(dirname "$PWD")"


nvidia-docker run -it \
    --env LICENSE=yes \
    -p 5555:5555 \
    -v $PWD:/home/jupyter/app \
    -v $PARENTDIR/data:/home/jupyter/data \
    -v $HOME/.vscode-server:/root/.vscode-server \
    -v /home/jupyter/app \
    tensorflow

# nvidia-docker run -it \
#     --env LICENSE=yes \
#     -p 5555:5555 \
#     -v $PWD:/home/jupyter/app \
#     -v $PARENTDIR/data:/home/jupyter/data \
#     -v $HOME/.vscode-server:/root/.vscode-server \
#     -v /home/jupyter/app \
#     temporal

# TODO: see how to use udi mapping with vscode
sudo chown -R $USER:$USER $PARENTDIR/data
sudo chown -R $USER:$USER $PWD
