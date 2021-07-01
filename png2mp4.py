#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
from PIL import Image


def main():
    dirname = os.path.join('/', 'Users', 'rapsealk', 'Desktop', 'Screenshots')
    files = os.listdir(dirname)
    files = sorted(files)
    print(files)

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # writer = cv2.VideoWriter('output.avi', fourcc, fps=10, frameSize=(792, 945))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter('output.mp4', fourcc, fps=10, frameSize=(792, 945))

    for file_ in files:
        image = Image.open(os.path.join(dirname, file_))
        image = np.asarray(image)[:, :, ::-1]
        writer.write(image)

    writer.release()


if __name__ == "__main__":
    main()
