import os
import PIL.Image as Image
import numpy as np


def areaCount(ids):
    imgPath = r"D:\data\huaxiSkin\0706opt_seg\JPEGImages"
    segPath = r"D:\data\huaxiSkin\opt_706_smpresNextUnet"
    for i in ids:
        fimg = os.path.join(imgPath, i + ".jpg")
        fseg = os.path.join(segPath, i + '.bmp')
        npseg = np.array(Image.open(fseg))
        hseg = np.sum(npseg == 120)
        qseg = np.sum(npseg == 240)
        print(f"img :{i}.jpg with hand {hseg}. and write {qseg}. and hand area {hseg / qseg * (7.0 * 7.5)} mm^2")


def tfunc():
    a = [1, 3, 4]
    b = [2, 6, 8]
    for x, y in zip(a, b):
        print(x, y)


if __name__ == '__main__':
    tfunc()
    # areaCount(['0706094233', '0706094259'])
