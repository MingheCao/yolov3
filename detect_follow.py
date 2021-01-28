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

import sys
sys.path.append('/usr/lib/python3/dist-packages')
import virtkey


def get_point(event, x, y, flags, param):
    global imgcor_x,imgcor_y,status
    if event == cv2.EVENT_LBUTTONDOWN:
        imgcor_x=x
        imgcor_y=y
        status=True
        v.press_unicode(ord('n'))
        v.release_unicode(ord('n'))

    if event == cv2.EVENT_RBUTTONDOWN:
        v.press_unicode(ord('n'))
        v.release_unicode(ord('n'))
        status=False

if __name__ == '__main__':
    check_requirements()

    source = '/home/rislab/Workspace/yolov3/data/images/*.jpg'
    outpath= '/home/rislab/Workspace/'
    weights = 'yolov3.pt'
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45
    device = select_device('')
    augment = False
    agnostic_nms = True
    classes = None

    imgcor_x=-1
    imgcor_y=-1
    status=False
    v = virtkey.virtkey()

    files = sorted(glob.glob(source))

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) if device.type != 'cpu' else None  # run once

    cv2.namedWindow("Follow Image", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    for idx,p in enumerate(files):
        img0 = cv2.imread(p)
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float().div(255).unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img0.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                bboxes=[]
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).round().tolist()  # normalized xywh
                    bboxes.append(xywh)

                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    # cv2.rectangle(im0,(round(xywh[0]-xywh[2]/2),round(xywh[1]-xywh[3]/2)),(round(xywh[0]+xywh[2]/2),round(xywh[1]+xywh[3]/2)),(0, 255, 0), 2)

                cv2.imshow('Follow Image', im0)
                cv2.setMouseCallback('Follow Image', get_point, im0)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                print('x:%d,y:%d' % (imgcor_x, imgcor_y))
                if status == True:
                    dists = []
                    for c in bboxes:
                        dist = pow((c[0] - imgcor_x), 2) + pow((c[1] - imgcor_y), 2)
                        dists.append(dist)
                    sort_index = np.argsort(np.array(dists))

                    gtbox = bboxes[sort_index[0]]

                    print('gtbox:', gtbox)
                    im2=img0.copy()
                    cv2.rectangle(im2, (round(gtbox[0] - gtbox[2] / 2), round(gtbox[1] - gtbox[3] / 2)),
                                  (round(gtbox[0] + gtbox[2] / 2), round(gtbox[1] + gtbox[3] / 2)), (0, 0, 255), 4)
                    # cv2.imwrite(outpath + str(idx) + '.jpg', im2)
                    cv2.imshow('Follow Image', im2)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                else:
                    im2 = img0.copy()
                    init_rect = cv2.selectROI('Follow Image', im2, True, False)
                    x, y, w, h = init_rect
                    gtbox = [x + w / 2, y + h / 2, w, h]
                    print(gtbox)
                    # cv2.imwrite(folderpath + str(idx) + '.jpg', cv_image)

                imgcor_x = -1
                imgcor_y = -1
