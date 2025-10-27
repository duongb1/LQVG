#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
from PIL import Image

import util
from util.transforms import letterbox
import datasets.transforms_image as T

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import random


def filelist(root, file_type):
    root = str(root)
    return [
        os.path.join(directory_path, f)
        for directory_path, directory_name, files in os.walk(root)
        for f in files if f.endswith(file_type)
    ]


class RSVGDataset(data.Dataset):
    """
    Dataset đọc ảnh + bbox + câu mô tả từ thư mục Annotations (VOC XML)
    và thư mục JPEGImages. File split (train/val/test).txt có thể là:
      - Danh sách số (chỉ số dựa theo count duyệt object như bản cũ)
      - Danh sách tên file (có/không có .jpg)
    """
    def __init__(self, images_path, anno_path, imsize=800, transform=None, augment=False,
                 split='train', testmode=False):
        self.images = []
        self.images_path = Path(images_path)
        self.anno_path = Path(anno_path)
        self.imsize = int(imsize)
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode

        # Đọc split
        split_path = self.images_path.parent / f"{self.split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"[RSVGDataset] Không tìm thấy file split: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            raw_lines = [x.strip() for x in f if x.strip()]

        # Xác định dạng split: số hay tên file
        is_all_numeric = all(line.isdigit() for line in raw_lines)

        # Chuẩn bị mapping filename (xml) -> sử dụng để so khớp khi split là tên
        # Trong VOC XML: <filename>abc.jpg</filename> (đôi khi không có đuôi)
        # Ta sẽ so sánh theo cả full name và stem để an toàn.
        wanted_numeric = set()
        wanted_names_full = set()
        wanted_names_stem = set()

        if is_all_numeric:
            wanted_numeric = set(int(x) for x in raw_lines)
        else:
            # Chuẩn hoá tên (lower)
            for x in raw_lines:
                name = x.lower()
                wanted_names_full.add(name)
                # nếu có .jpg/.png thì lấy stem, nếu không thì vẫn add chính nó như stem
                stem = Path(name).stem
                wanted_names_stem.add(stem)

        # Duyệt toàn bộ annotation xml theo thứ tự ổn định
        annotations = sorted(filelist(self.anno_path, '.xml'))
        if len(annotations) == 0:
            raise RuntimeError(f"[RSVGDataset] Không tìm thấy file XML trong: {self.anno_path}")

        count = 0
        miss_img_files = 0
        chosen_objects = 0

        for anno_fp in annotations:
            try:
                root = ET.parse(anno_fp).getroot()
            except Exception as e:
                print(f"[RSVGDataset] Cảnh báo: lỗi đọc XML {anno_fp}: {e}")
                continue

            # Lấy tên file từ XML
            xml_fname = root.findtext("./filename", default="").strip()
            xml_fname_lower = xml_fname.lower()

            # Có trường hợp filename trong XML không có đuôi → suy diễn .jpg
            # Ta sẽ thử các phương án: tên như trong XML, thêm .jpg, thêm .png
            candidates = []
            if xml_fname_lower:
                candidates.append(xml_fname_lower)
                stem = Path(xml_fname_lower).stem
                if "." not in Path(xml_fname_lower).name:
                    candidates.append(f"{xml_fname_lower}.jpg")
                    candidates.append(f"{xml_fname_lower}.png")
                # cũng lưu stem để so khớp split dạng không đuôi
                candidates.append(stem)

            # Xác định có lấy ảnh này theo split không
            take_this_image = False
            if is_all_numeric:
                # Giữ logic cũ: nếu split là số thì so theo count từng object
                # (chọn ở cấp object ngay bên dưới)
                pass
            else:
                # split là theo tên: nếu bất kỳ candidate trùng với full list hoặc stem list → lấy
                for c in candidates:
                    c = c.lower()
                    if c in wanted_names_full or Path(c).stem in wanted_names_stem:
                        take_this_image = True
                        break

            # Lấy path ảnh thật
            # Ưu tiên đúng tên từ XML; nếu thiếu đuôi, thử .jpg/.png
            img_path = None
            for c in candidates:
                c_name = Path(c).name  # chỉ lấy tên file
                trial = self.images_path / c_name
                if trial.exists():
                    img_path = trial
                    break
            if img_path is None and xml_fname_lower:
                # last resort: lấy đúng như XML ghép thẳng
                trial = self.images_path / xml_fname
                if trial.exists():
                    img_path = trial

            # Nếu vẫn không có file ảnh, cảnh báo và bỏ qua object của ảnh này
            if img_path is None:
                miss_img_files += 1
                # print(f"[RSVGDataset] Cảnh báo: không tìm thấy ảnh cho XML {anno_fp} (filename='{xml_fname}')")
                continue

            # Duyệt từng object trong ảnh
            for member in root.findall('object'):
                # Nếu split là số → dùng count như logic cũ
                if is_all_numeric:
                    if count in wanted_numeric:
                        # lấy object này
                        pass
                    else:
                        count += 1
                        continue
                else:
                    # split là tên: chỉ lấy nếu ảnh thuộc split
                    if not take_this_image:
                        continue

                # Đọc bbox (VOC: xmin, ymin, xmax, ymax)
                try:
                    bnd = member.find('bndbox')
                    xmin = int(float(bnd.find('xmin').text))
                    ymin = int(float(bnd.find('ymin').text))
                    xmax = int(float(bnd.find('xmax').text))
                    ymax = int(float(bnd.find('ymax').text))
                    box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                except Exception as e:
                    # Bbox hỏng → bỏ qua object
                    # print(f"[RSVGDataset] Cảnh báo bbox lỗi tại {anno_fp}: {e}")
                    if is_all_numeric:
                        count += 1
                    continue

                # Lấy câu mô tả (nếu không có, dùng nhãn class)
                # Trong VOC thường: <object><name>class</name> ...><attributes/mô tả khác...>
                # Ở code cũ dùng member[3].text → rất dễ lệch nếu XML khác cấu trúc.
                # Ta lấy 'name' làm fallback và thử 'description' nếu có.
                text_node = member.find('description')
                if text_node is not None and text_node.text and text_node.text.strip():
                    text = text_node.text.strip()
                else:
                    name_node = member.find('name')
                    text = name_node.text.strip() if (name_node is not None and name_node.text) else "object"

                self.images.append((str(img_path), box, text))
                chosen_objects += 1

                if is_all_numeric:
                    count += 1  # chỉ tăng khi dùng logic numeric

        if chosen_objects == 0:
            raise RuntimeError(
                f"[RSVGDataset] Không lấy được object nào. Kiểm tra split ({split_path}), "
                f"filename trong XML, và thư mục ảnh {self.images_path}"
            )
        if miss_img_files > 0:
            print(f"[RSVGDataset] Cảnh báo: {miss_img_files} ảnh không tìm thấy, đã bỏ qua.")

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)  # x1 y1 x2 y2
        # Đọc ảnh bằng PIL (RGB)
        img = Image.open(img_path).convert('RGB')
        return img, phrase, bbox, img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_path = self.pull_item(idx)
        caption = " ".join(str(phrase).lower().split())

        # Lấy kích thước gốc (PIL: size = (w, h))
        w, h = img.size

        if self.testmode:
            # letterbox cần numpy array
            img_np = np.array(img)  # HWC, RGB
            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            img_np, mask, ratio, dw, dh = letterbox(img_np, mask, self.imsize)

            # scale bbox theo ratio + padding
            bbox = bbox.astype(np.float32)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh

            # chuyển lại về PIL để đi qua pipeline transform (nếu transform chấp nhận PIL/numpy đều OK)
            img = Image.fromarray(img_np)

        # torchize bbox
        bbox = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=torch.float32).unsqueeze(0)

        target = {
            "dataset_name": "RSVG",
            "boxes": bbox,
            "labels": torch.tensor([1], dtype=torch.long),
            "valid": torch.tensor([1], dtype=torch.long),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        if caption is not None:
            target["caption"] = caption

        # Transform (ToTensor + Normalize)
        if self.transform is not None:
            img, target = self.transform(img, target)

        # Trả về dạng [1, 3, H, W] (T=1)
        if self.testmode:
            return img.unsqueeze(0), target, dw, dh, img_path, ratio
        else:
            return img.unsqueeze(0), target


def make_coco_transforms(image_set, cautious):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 560, 640, 720, 800]
    max_size = 800

    if image_set == "train":
        return T.Compose([
            T.RandomResize(scales, max_size=max_size),
            normalize
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])


def build(image_set, args):
    root = Path(args.rsvg_path)
    assert root.exists(), f'provided RSVG path {root} does not exist'

    img_folder = root / "JPEGImages"
    ann_folder = root / "Annotations"

    dataset = RSVGDataset(
        img_folder,
        ann_folder,
        transform=make_coco_transforms(image_set, False),
        split=image_set,
        testmode=(image_set == 'test')
    )
    return dataset
