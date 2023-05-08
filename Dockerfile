FROM pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9

SHELL ["/bin/bash", "-c"]

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python -m pip install --no-cache-dir --upgrade pip==22.3.1 \
    && pip install --root-user-action=ignore --no-cache-dir --default-timeout=900 \
    opencv-python-headless==4.5.5.62 \
    einops \
    shapely \
    timm \
    yacs \
    tensorboardX \
    ftfy \
    prettytable \
    pymongo \
    transformers \
    pycocotools  \
    matplotlib \
    onnxruntime \
    onnx \
    && pip cache purge


RUN git clone https://github.com/microsoft/GLIP.git \
    && cd GLIP \
    && python setup.py build develop --user \
    && cd .. \
    && mkdir glip-models


RUN git clone https://github.com/facebookresearch/segment-anything.git \
    && cd segment-anything \
    && pip install -e . \
    && cd .. \
    && mkdir sam-models


RUN wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth  -O glip-models/glip_tiny_model_o365_goldg_cc_sbu.pth \
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam-models/sam_vit_h_4b8939.pth

# This is just to have less stuff to look at.
RUN rm -rf NVIDIA_Deep_Learning_Container_License.pdf README.md docker-examples examples tutorials

CMD ["bash"]
