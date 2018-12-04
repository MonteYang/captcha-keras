# -*- coding: utf-8 -*-
import os


def mkdir(path):
    """
    创建目录
    :param path:
    :return:
    """
    path = path.strip()
    path = path.rstrip("\\")
    path = path.rstrip("/")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        print(str(path) + " exists")
        return False
