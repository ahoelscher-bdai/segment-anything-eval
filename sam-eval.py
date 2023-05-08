import torch
import torchvision
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import requests
from io import BytesIO
from PIL import Image
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

GLIP_WEIGHT_FILE = "/workspace/glip-models/glip_tiny_model_o365_goldg_cc_sbu.pth"
GLIP_CONFIG_FILE = "/workspace/GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"


def get_glip():
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(GLIP_CONFIG_FILE)
    cfg.merge_from_list(["MODEL.WEIGHT", GLIP_WEIGHT_FILE])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    return GLIPDemo(
        cfg,
        min_image_size=480,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )


def main():
    image_dir = "/workspace/sam-eval/external/bdai-data/projects/watch_understand_do/diagnose_repair/locate_tool"

    glip = get_glip()


if __name__ == "__main__":
    main()
