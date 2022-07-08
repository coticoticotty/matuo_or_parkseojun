import glob
import os
import sys

def main(dir_name):
    target_extension = [".jpg"]

    # 指定したディレクトリのファイル
    files = glob.glob(f"./{dir_name}/*")

    for index, file in enumerate(files):
        path, extension = os.path.splitext(file)

        if not extension in target_extension:
            print(f'Delete {path}{extension}')
            os.remove(file)
        else:
            print(f"Don't Delete {path}{extension}")

if __name__ == '__main__':
    dir_name = sys.argv[1]
    main(dir_name)