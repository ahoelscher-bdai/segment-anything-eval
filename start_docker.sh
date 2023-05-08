#!/bin/bash

nvidia-docker run --gpus all --rm -it -v.:/workspace/sam-eval --entrypoint bash sam-eval
