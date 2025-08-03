from typing import Dict
from ultralytics import YOLO

class Yolov11Detector:
    def __init__(self, detection_cfg: Dict, crop_roi=None):
        self.__model = YOLO(detection_cfg["model_path"], task=detection_cfg["task"])
        self.__imgsz = detection_cfg["imgsz"]
        self.__conf = detection_cfg["conf_thresh"]
        self.__iou = detection_cfg["iou_thresh"]
        self.__device = detection_cfg["device"]
        self.__classes = detection_cfg["classes"]
        self.__max_det = detection_cfg["max_det"]
        self.__min_obj_w = detection_cfg["min_obj_w"]
        self.__min_obj_h = detection_cfg["min_obj_h"]
        self.__crop_roi = crop_roi
    
    def _preprocess(self, imgs_bgr):
        return imgs_bgr
    
    def detect(self, imgs_bgr):
        imgs = self._preprocess(imgs_bgr)
        predicts = self.__model(
            source=imgs,
            imgsz=self.__imgsz,
            conf=self.__conf,
            iou=self.__iou,
            device=self.__device,
            classes=self.__classes,
            max_det=self.__max_det,
            verbose=False
        )
        return self._postprocess(predicts)
    
    def _postprocess(self, preds):
        l_boxes = []
        for pred in preds:
            boxes = []
            for box in pred.boxes.data:
                x1, y1, x2, y2, sc, cl = box.cpu().numpy().tolist()
                w = (x2 - x1)
                h = (y2 - y1)
                if w < self.__min_obj_w: continue
                if h < self.__min_obj_h: continue
                if self.__crop_roi is not None:
                    x1 = x1 + self.__crop_roi[0]
                    y1 = y1 + self.__crop_roi[2]
                    x2 = x2 + self.__crop_roi[0]
                    y2 = y2 + self.__crop_roi[2]
                l_boxes.append([x1, y1, x2, y2, sc])
                # l_boxes.append([x1, y1, x2, y2])
        return l_boxes