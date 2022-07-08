import glob
from PIL import Image
import os
import sys

import cv2
import torch
import torch.nn as nn
from torchvision import transforms

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


image_size = 224

def predict_who_all():
    classes = ["Matsuo", "ParkSeroi"]
    target_path = os.path.join('./test_data' + '/*.jpg')
    files = glob.glob(target_path)

    face_cascade_path = 'c:/users/kohei okamoto/anaconda3/envs/face_recognition/lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
    # 顔検出器の読み込み
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for file_path in files:
        file_name = os.path.basename(file_path)
        # 画像の読み込み
        image = cv2.imread(file_path)
        is_grayscale = (image.shape[2] == 1)

        # 白黒画像の場合はRGBに変換
        if is_grayscale == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # openCVで顔検出をする
        faces = face_cascade.detectMultiScale(image_gray)

        if len(faces) > 0:
            for index, (x, y, w, h) in enumerate(faces, 1):
                face = image[y - 15:y + h + 15, x - 10:x + w + 10]
                face = cv2.resize(face, (image_size, image_size))
                face = Image.fromarray(face)  # torchvisionのtransformに渡すにはPILイメージに変換する必要がある。
                mean = IMAGENET_DEFAULT_MEAN
                std = IMAGENET_DEFAULT_STD
                transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
                face = transform(face)
                face = face.unsqueeze_(0)

                # 学習済みのVision Transferモデルをロード
                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

                # 画像を学習したパラメータの読み込み
                model.load_state_dict(torch.load('./pytorch_vit_transfered_model.pth'))

                model.eval()
                outputs = model(face)

                # ソフトマックス関数で確率に変換
                softmax = torch.nn.Softmax(dim=1)
                softmax_result = softmax(outputs)

                value, pred_class = torch.max(softmax_result, 1)

                predict_name = classes[pred_class.item()]

                label = f'{predict_name}({value.item() * 100:.1f}%)'

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)  # 四角形描画
                cv2.putText(image, label, (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 0, 0), 3)  # 人物名記述

            save_dir = 'test_data/detected_images/'
            save_path = os.path.join(save_dir + file_name)
            cv2.imwrite(save_path, image)

        else:
            print('no_face')

def predict_who(file_name):
    classes = ["Matsuo", "ParkSeroi"]
    file_name = file_name
    file_path = os.path.join('./test_data/' + file_name)

    face_cascade_path = 'c:/users/kohei okamoto/anaconda3/envs/face_recognition/lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
    # 顔検出器の読み込み
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # 画像の読み込み
    image = cv2.imread(file_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # openCVで顔検出をする
    faces = face_cascade.detectMultiScale(image_gray)

    if len(faces) > 0:
        for index, (x, y, w, h) in enumerate(faces, 1):
            face = image[y - 15:y + h + 15, x - 10:x + w + 10]
            face = cv2.resize(face, (image_size, image_size))
            face = Image.fromarray(face)  # torchvisionのtransformに渡すにはPILイメージに変換する必要がある。
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            face = transform(face)
            face = face.unsqueeze_(0)

            # 学習済みのVision Transferモデルをロード
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

            # 画像を学習したパラメータの読み込み
            model.load_state_dict(torch.load('./pytorch_vit_transfered_model.pth'))

            model.eval()
            outputs = model(face)

            # ソフトマックス関数で確率に変換
            softmax = torch.nn.Softmax(dim=1)
            softmax_result = softmax(outputs)

            value, pred_class = torch.max(softmax_result, 1)

            predict_name = classes[pred_class.item()]

            label = f'{predict_name}({value.item()*100:.1f}%)'

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)  # 四角形描画
            cv2.putText(image, label, (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 0, 0), 3)  # 人物名記述

        save_dir = 'test_data/detected_images/'
        save_path = os.path.join(save_dir + file_name)
        cv2.imwrite(save_path, image)

    else:
        print('no_face')

if __name__ == '__main__':
    predict_who_all()