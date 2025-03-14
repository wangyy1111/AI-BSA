import os
import os.path as osp
import glob
import PIL.Image as Image

def renameAndSave():
    """one time use."""
    iptPath = r"C:\Users\10911\Desktop\华西皮肤\0706"
    optPath = r"C:\Users\10911\Desktop\华西皮肤\0706opt"
    if not osp.exists(optPath):
        os.makedirs(optPath)

    fl = glob.glob(osp.join(iptPath, "*.*"))
    for f in fl:
        bname = osp.basename(f)
        pimg = Image.open(f)
        pimg.save(osp.join(optPath, bname[len("微信图片_2020"):]))

        print(f)


if __name__ == '__main__':
    renameAndSave()
