import glob
import os
from pathlib import Path

import cv2

def main():
    source_dir_name = 'matsuoshun'
    detect_face_and_save_image(source_dir_name)

def detect_face_and_save_image(source_dir):
    current_dir = Path(__file__).parent

    face_cascade_path = os.path.join(current_dir,'cascade', 'haarcascade_frontalface_alt.xml')
    # 顔検出器の読み込み
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # 指定したディレクトリのファイル
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = os.path.join(project_dir, 'data')

    files = glob.glob(f"{data_dir}/{source_dir}/*")

    for index, file in enumerate(files):
        file_name, _ = os.path.splitext(os.path.basename(file))

        # 画像の読み込み
        src = cv2.imread(file)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # openCVで顔検出をする
        faces = face_cascade.detectMultiScale(src_gray)

        for index, (x, y, w, h) in enumerate(faces, 1):
            face = src[y - 15:y + h + 15, x - 10:x + w + 10]
            try:
                save_path = os.path.join(data_dir, 'face_detected', f'{file_name}_{index}.jpg')
                cv2.imwrite(save_path, face)
            except:
                print(f'失敗')

        if index % 10 == 0:
            print(f'{index*1}枚検出完了')

if __name__ == '__main__':
    main()



