# -*- coding: utf-8 -*-

import os
import shutil


def remove_big_file(root_dir):
    for name in os.listdir(root_dir):
        file = os.path.join(root_dir, name)
        if os.path.isfile(file):
            try:
                file_size = os.path.getsize(file)  # 字节
                if file_size > 200 * 1024 * 1024:  # 3M
                    print(f"删除文件:{file}")
                    os.remove(file)
                if file.endswith(".html"):
                    if '_1.html' in file or '_700.html' in file:
                        continue
                    os.remove(file)
                # predictions_valid_epoch_697.json
                if name.startswith("predictions_") and "_epoch_" in name and name.endswith(".json"):
                    if '_1.json' in file or '_700.json' in file:
                        continue
                    print(f"删除文件:{file}")
                    os.remove(file)
            except Exception as e:
                print(f"异常:{e}")
        elif os.path.isdir(file):
            _dir_name = os.path.basename(file)
            if _dir_name.startswith(".") or "__pycache__".__eq__(_dir_name):
                print(f"删除整个文件夹:{file}")
                try:
                    shutil.rmtree(file)
                except Exception as e:
                    print(f"异常:{e}")
            else:
                remove_big_file(file)


if __name__ == '__main__':
    remove_big_file(
        root_dir=r"D:\工作\授课\2025\05_NLP\20260201"
        # root_dir=r"."
    )
