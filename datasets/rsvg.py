#!/usr/bin/env python
# -*-coding: utf -8-*-
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import util
from util.transforms import letterbox
# from torchvision.transforms import Compose, ToTensor, Normalize
import datasets.transforms_image as T
import matplotlib.pyplot as plt
import torch.utils.data as data
import random
import torch
from pathlib import Path


def generate_gaussian_heatmap(box, size=32, sigma=1.5):
    """Create a gaussian heatmap centered at the normalized box center."""
    cx, cy, _, _ = box
    cx = np.clip(cx * size, 0, size - 1)
    cy = np.clip(cy * size, 0, size - 1)

    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heatmap = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=800, transform= None, augment= False,
                 split='train', testmode=False):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        # self.query_len = max_query_len  # 40
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        root = Path(self.images_path).parent
        file = open(root / f"{self.split}.txt", "r", encoding="utf-8").readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        annotations = filelist(anno_path, '.xml')
        for anno_path in annotations:
            root = ET.parse(anno_path).getroot()
            for member in root.findall('object'):
                if count in Index:
                    imageFile = str(images_path) + '/' + root.find("./filename").text
                    box = np.array([int(member[2][0].text), int(member[2][1].text), int(member[2][2].text), int(member[2][3].text)],dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        # bbox =torch.tensor(bbox).to(torch.float)
        bbox = np.array(bbox, dtype=int)  # box format: to x1 y1 x2 y2
        # img_bgr = cv2.imread(img_path)
        # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        # img = np.array(img)

        return img, phrase, bbox, img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_path  = self.pull_item(idx)
        # phrase = phrase.lower()
        caption = " ".join(phrase.lower().split())

        # seems a bug in torch transformation resize, so separate in advance
        # h, w = img.shape[0], img.shape[1]
        w, h = img.size
        mask = np.zeros_like(img)

        if self.testmode:
            img = np.array(img)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
        bbox = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]]).to(torch.float)
        bbox = bbox.unsqueeze(0)

        target = {}
        target["dataset_name"] = "RSVG"
        target["boxes"] = bbox
        target["labels"] = torch.tensor([1])
        if caption is not None:
            target["caption"] = caption
        # target["image_id"] = image_id
        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # Norm, to tensor
        if self.transform is not None:
            img, target = self.transform(img, target)

        boxes = target.get("boxes")
        valids = target.get("valid")
        if boxes is not None and valids is not None:
            heatmap_size = 32
            heatmap_tensor = torch.zeros((boxes.shape[0], heatmap_size, heatmap_size), dtype=torch.float32)
            for idx, (box, valid_flag) in enumerate(zip(boxes, valids)):
                if valid_flag.item() > 0:
                    heatmap_np = generate_gaussian_heatmap(box.cpu().numpy(), size=heatmap_size)
                    heatmap_tensor[idx] = torch.from_numpy(heatmap_np)
            target["heatmap"] = heatmap_tensor
        else:
            target["heatmap"] = torch.zeros((0, 32, 32), dtype=torch.float32)

        if self.testmode:
            return img.unsqueeze(0), target, dw, dh, img_path, ratio
        else:
            return img.unsqueeze(0), target
        # return img: [1, 3, H, W], the first dimension means T = 1.


def make_coco_transforms(image_set, cautious):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 560, 640, 720, 800]
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    # final_scales = [296, 328, 360, 392, 416, 448, 480, 512]

    max_size = 800
    if image_set == "train":
        return T.Compose(
            [T.RandomResize(scales, max_size=max_size),
            normalize]
        )

    else:
        return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


    raise ValueError(f"unknown {image_set}")

from pathlib import Path

def build(image_set, args):
    root = Path(args.rsvg_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    input_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    img_folder = root / "JPEGImages"
    ann_file = root / "Annotations"

    dataset = RSVGDataset(img_folder, ann_file, transform=make_coco_transforms(image_set, False), split=image_set, testmode=(image_set=='test'))
    return dataset


