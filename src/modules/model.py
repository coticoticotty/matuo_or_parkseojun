import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

IMAGE_SIZE = 224

def predict(pil_image):
    classes = ["Matsuo", "ParkSeroi"]

    image_as_numpy = np.array(pil_image, dtype=np.uint8)
    # PILImageを openCV用に変更
    cv2_image = cv2.cvtColor(image_as_numpy, cv2.COLOR_BGR2RGB)

    image_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    root_dir = Path(__file__).resolve().parent.parent.parent
    face_cascade_path = os.path.join(root_dir, 'script', 'cascade', 'haarcascade_frontalface_alt.xml')
    # 顔検出器の読み込み
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # openCVで顔検出をする
    faces = face_cascade.detectMultiScale(image_gray)

    if len(faces) > 0:
        for index, (x, y, w, h) in enumerate(faces, 1):
            face = cv2_image[y - 15:y + h + 15, x - 10:x + w + 10]
            face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)  # torchvisionのtransformに渡すにはPILイメージに変換する必要がある。
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            face = transform(face)
            face = face.unsqueeze_(0)

            # 学習済みのVision Transferモデルをロード
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

            # 画像を学習したパラメータの読み込み
            weight_file_path = os.path.join(root_dir, 'script', 'pytorch_vit_transfered_model.pth')
            model.load_state_dict(torch.load(weight_file_path))

            model.eval()
            outputs = model(face)

            # ソフトマックス関数で確率に変換
            softmax = torch.nn.Softmax(dim=1)
            softmax_result = softmax(outputs)

            value, pred_class = torch.max(softmax_result, 1)

            predict_name = classes[pred_class.item()]

            label = f'{predict_name}({value.item() * 100:.1f}%)'

            cv2.rectangle(cv2_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)  # 四角形描画
            cv2.putText(cv2_image, label, (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 0, 0), 2)  # 人物名記述

        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        return pil_image
