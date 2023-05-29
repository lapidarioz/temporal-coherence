#!/usr/bin/env bash

PARENTDIR="$(dirname "$PWD")"

echo "---
version: \"3\"
services:
  tensorflow:
    build: 
      context: .
      dockerfile: Dockerfile.tensorflow
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    volumes:
      - $PARENTDIR/data:/home/jupyter/data
      - $PWD:/home/jupyter/app
      - $HOME/.vscode-server:/.vscode-server
    ports:
      - 5555:5555
    user: $(id -u):$(id -g)" > $PWD/docker-compose.yml


docker compose build


