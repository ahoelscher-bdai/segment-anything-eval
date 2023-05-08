FROM pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9

SHELL ["/bin/bash", "-c"]

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install packages inside the new environment
RUN python -m pip install --no-cache-dir --upgrade pip==22.3.1 \
    && pip install --root-user-action=ignore --no-cache-dir --default-timeout=900 \
    einops \
    shapely \
    timm \
    yacs \
    tensorboardX \
    ftfy \
    prettytable \
    pymongo \
    transformers \
    opencv-python \
    pycocotools  \
    matplotlib \
    onnxruntime \
    onnx \
    && pip cache purge


RUN git clone https://github.com/microsoft/GLIP.git \
    && cd GLIP \
    && python setup.py build develop --user \
    && cd .. \
    && mkdir glip-models \
    && cd glip-models \
    && wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/swin_tiny_patch4_window7_224.pth -O swin_tiny_patch4_window7_224.pth \
    && wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/swin_large_patch4_window12_384_22k.pth -O swin_large_patch4_window12_384_22k.pth \


RUN git clone git@github.com:facebookresearch/segment-anything.git \
    && cd segment-anything \
    && pip install -e . \
    && mkdir sam-models \
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \

CMD ["bash"]
