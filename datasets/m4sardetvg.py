#!/usr/bin/env python
# -*-coding: utf-8-*-
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.utils.data as data
import datasets.transforms_image as T
from util.transforms import letterbox

def filelist(root, file_type):
    return [
        os.path.join(dp, f)
        for dp, _, files in os.walk(root)
        for f in files if f.endswith(file_type)
    ]

def _is_number(s: str) -> bool:
    if s is None: return False
    s = s.strip()
    if s == "": return False
    # chấp nhận số âm và float
    if s[0] in "+-":
        s2 = s[1:]
    else:
        s2 = s
    # chỉ một dấu chấm
    parts = s2.split(".")
    if len(parts) > 2: return False
    return all(p.isdigit() for p in parts if p != "") and (s2.replace(".", "").isdigit())

class M4SARDetVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=512, transform=None, augment=False,
                 split='train', testmode=False):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode

        # Đọc split list, hỗ trợ "00001" / "1" / "00001.jpg"
        root_dir = Path(self.images_path).parent
        lines = (root_dir / f"{self.split}.txt").read_text(encoding="utf-8").splitlines()

        def _norm(s: str) -> str:
            s = s.strip()
            s = s.split('.')[0]  # "00001.jpg" -> "00001"
            return s

        index_str_set = set(_norm(s) for s in lines)
        index_nozero_set = set(s.lstrip('0') or '0' for s in index_str_set)
        max_len = max((len(s) for s in index_str_set), default=1)

        def in_index(cnt: int) -> bool:
            s = str(cnt)
            return (s in index_str_set) or (s in index_nozero_set) or (s.zfill(max_len) in index_str_set)

        count = 0
        annotations = sorted(filelist(anno_path, '.xml'))

        for anno_p in annotations:
            root = ET.parse(anno_p).getroot()

            # Lấy filename
            fn_node = root.find("filename")
            if fn_node is None or not fn_node.text: 
                continue
            image_file = str(images_path) + '/' + fn_node.text.strip()

            # Lấy size để clip bbox (nếu có)
            W = H = None
            size_node = root.find("size")
            if size_node is not None:
                w_node = size_node.find("width")
                h_node = size_node.find("height")
                if w_node is not None and h_node is not None and _is_number(w_node.text) and _is_number(h_node.text):
                    W = int(float(w_node.text))
                    H = int(float(h_node.text))

            for member in root.findall('object'):
                if not in_index(count):
                    count += 1
                    continue

                # --- BBOX ---
                bnd = member.find('bndbox')
                if bnd is None:
                    count += 1
                    continue

                xmin_node = bnd.find('xmin'); ymin_node = bnd.find('ymin')
                xmax_node = bnd.find('xmax'); ymax_node = bnd.find('ymax')
                if (xmin_node is None or ymin_node is None or xmax_node is None or ymax_node is None):
                    count += 1
                    continue
                if not (_is_number(xmin_node.text) and _is_number(ymin_node.text) and
                        _is_number(xmax_node.text) and _is_number(ymax_node.text)):
                    count += 1
                    continue

                xmin = float(xmin_node.text); ymin = float(ymin_node.text)
                xmax = float(xmax_node.text); ymax = float(ymax_node.text)

                if xmax < xmin: xmin, xmax = xmax, xmin
                if ymax < ymin: ymin, ymax = ymax, ymin

                # clip theo size nếu có
                if W is not None and H is not None:
                    xmin = min(max(0.0, xmin), W - 1)
                    xmax = min(max(0.0, xmax), W - 1)
                    ymin = min(max(0.0, ymin), H - 1)
                    ymax = min(max(0.0, ymax), H - 1)

                if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                    count += 1
                    continue

                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

                # --- DESCRIPTION (referring expression) ---
                desc_node = member.find('description')
                if desc_node is not None and desc_node.text and desc_node.text.strip() != "":
                    phrase = desc_node.text.strip()
                else:
                    # fallback nhẹ sang class name
                    name_node = member.find('name')
                    if name_node is None or name_node.text is None or name_node.text.strip() == "":
                        count += 1
                        continue
                    phrase = name_node.text.strip()

                self.images.append((image_file, box, phrase))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)  # xyxy

        # SAR 1 kênh -> RGB 3 kênh
        imgL = Image.open(img_path).convert('L')
        img  = Image.merge('RGB', (imgL, imgL, imgL))

        return img, phrase, bbox, img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_path = self.pull_item(idx)
        caption = " ".join(phrase.lower().split())

        # w,h theo PIL
        w, h = img.size
        if self.testmode:
            im_np = np.array(img)                 # HWC, uint8
            mask  = np.zeros_like(im_np)          # mask giả để gọi letterbox
            im_np, mask, ratio, dw, dh = letterbox(im_np, mask, self.imsize)
            # cập nhật bbox theo letterbox
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
            img = Image.fromarray(im_np)

        # target
        bbox_t = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=torch.float32).unsqueeze(0)
        target = {
            "dataset_name": "m4sardetvg",
            "boxes": bbox_t,
            "labels": torch.tensor([1]),
            "caption": caption,
            "valid": torch.tensor([1]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }

        # transforms
        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.testmode:
            # trả thêm dw, dh, img_path, ratio để hậu xử lý nếu cần
            return img.unsqueeze(0), target, dw, dh, img_path, ratio
        else:
            return img.unsqueeze(0), target

def make_coco_transforms(image_set, cautious):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [224, 256, 336, 432, 512]
    max_size = 512
    if image_set == "train":
        return T.Compose([T.RandomResize(scales, max_size=max_size), normalize])
    else:
        return T.Compose([T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

def build(image_set, args):
    root = Path(args.m4sardetvg_path)
    assert root.exists(), f'provided M4SARDetVG path {root} does not exist'

    img_folder = root / "JPEGImages"
    ann_folder = root / "Annotations"

    dataset = M4SARDetVGDataset(
        img_folder,
        ann_folder,
        transform=make_coco_transforms(image_set, False),
        split=image_set,
        testmode=(image_set == 'test')
    )
    return dataset
