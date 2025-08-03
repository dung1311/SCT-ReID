from yacs.config import CfgNode
import yaml
import cv2
import torch
import torch.nn.functional as F

from .engine import DefaultPredictor
from .config import get_cfg
from ..base import IEmbedding


class Embedding(IEmbedding):
    def __init__(self, cfg_dict):
        """
        Args:
            cfg_dict: dict hoặc string (đường dẫn tới file YAML của FastReID)
        """
        if isinstance(cfg_dict, str):
            print(f"[Embedding] Loading config from {cfg_dict}")
            with open(cfg_dict, "r") as f:
                cfg_dict = yaml.safe_load(f)

        self.cfg = get_cfg()
        self.cfg.merge_from_other_cfg(CfgNode(cfg_dict))
        self.predictor = DefaultPredictor(self.cfg)

    def _preprocess(self, imgs_bgr):
        if not isinstance(imgs_bgr, list): 
            imgs_bgr = [imgs_bgr]

        images = []
        for img_bgr in imgs_bgr:
            img_rgb = img_bgr[:, :, ::-1]
            image = cv2.resize(
                img_rgb,
                tuple(self.cfg.INPUT.SIZE_TEST[::-1]),
                interpolation=cv2.INTER_CUBIC
            )
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
            images.append(image)
            
        imgs = torch.cat(images, dim=0)
        return imgs
    
    def extract_feature(self, imgs_bgr):
        images = self._preprocess(imgs_bgr)
        features = self.predictor(images)
        features = F.normalize(features)
        return features.cpu().numpy()
