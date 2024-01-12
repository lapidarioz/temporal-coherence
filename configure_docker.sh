#!/usr/bin/env bash

PARENTDIR="$(dirname "$PWD")"

docker network create -d bridge jupyter

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
      - $HOME/.vscode-server-tensorflow:/root/.vscode-server
      - $PWD/code-server.yml:/root/.config/code-server/config.yaml
      - $HOME/.gitconfig:/etc/gitconfig
      - $HOME/.local/share/code-server/extensions:/root/.local/share/code-server/extensions
    ports:
      - 5555:5555
      - 7458:7458
    networks:
      - network1
   
networks:
  network1:
    name: jupyter
    external: true" > $PWD/docker-compose.yml


docker compose build


