import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np
import glob

import json

# using yolov3


def main(dataset, weight):
    files = sorted(glob.glob(dataset + '/*.jpg'))

    conf_thres = 0.25
    iou_thres = 0.45
    device = select_device('')
    augment = False
    agnostic_nms = True
    classes = None

    imgsz = 640

    model = attempt_load(weight, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    # Load model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) if device.type != 'cpu' else None  # run once

    detect_info = {}
    for idx, frame in enumerate(files):
        img0 = cv2.imread(frame)
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float().div(255).unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        detect_bboxes = []
        for i, det in enumerate(pred):  # detections per image
            im0 = img0.copy()
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                            ).view(-1).round().tolist()  # normalized xywh
                    bboxes = {}
                    bboxes['bbox'] = xywh
                    bboxes['conf'] = float(conf.cpu())
                    bboxes['cls'] = names[int(cls)]

                    detect_bboxes.append(bboxes)

        detect_info['/'.join(frame.split('/')[-2:])] = detect_bboxes

    with open(dataset + '/detections.json', 'w') as json_file:
        json.dump(detect_info, json_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='yolov3 person detection')
    parser.add_argument('--dataset', type=str, help='datasets')
    parser.add_argument('--weight', type=str, help='yolo weights')
    args = parser.parse_args()

    check_requirements()

    main(args.dataset, args.weight)
