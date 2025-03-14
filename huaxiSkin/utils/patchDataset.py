
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import os.path as osp
import os


# random.seed(666)


class PatchDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=(512, 512), patchSize=(512, 512)):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.patchSize = patchSize

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.seq = iaa.Sequential([
            # iaa.Dropout([0.05, 0.1]),  # drop 5% or 20% of all pixels
            #             iaa.AverageBlur(),
            iaa.HorizontalFlip(),
            iaa.VerticalFlip(),
            iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            iaa.Affine(scale=(0.8, 1.2), rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.ElasticTransformation(alpha=20, sigma=5)  # apply water effect (affects segmaps)
        ], random_order=False)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocessMask(cls, pil_mask, scale):
        w, h = pil_mask.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_mask = pil_mask.resize((newW, newH))

        img_nd = np.array(pil_mask)

        img_nd[img_nd == 120] = 1
        img_nd[img_nd == 240] = 2

        return img_nd

    def preprocessAll(self, pil_img, pil_mask, scale):
        assert pil_img.size == pil_mask.size

        # w, h = pil_mask.size
        newW, newH = scale[0], scale[1]
        pil_img = pil_img.resize((newW, newH))
        pil_mask = pil_mask.resize((newW, newH))

        np_img = np.array(pil_img)
        np_mask = np.array(pil_mask)

        if len(np_mask.shape) == 3:
            np_mask = np_mask[:, :, 0]

        if len(np_mask.shape) == 2:
            np_mask = np_mask[:, :, np.newaxis]

        np_mask[np_mask == 120] = 1
        np_mask[np_mask == 240] = 2

        np_mask = SegmentationMapsOnImage(np_mask, shape=np_img.shape)

        np_img, np_mask = self.seq(image=np_img, segmentation_maps=np_mask)

        np_mask = np_mask.get_arr()
        bplocal = np.where(np_mask != 0)
        loc = random.randint(0, len(bplocal[0]) - 1)
        locx = bplocal[0][loc]
        locy = bplocal[1][loc]
        if locx < self.patchSize[0] // 2:
            locx = self.patchSize[0] // 2
        if locx > newH - self.patchSize[0] // 2:
            locx = newH - self.patchSize[0] // 2

        if locy < self.patchSize[1] // 2:
            locy = self.patchSize[1] // 2
        if locy > newW - self.patchSize[1] // 2:
            locy = newW - self.patchSize[1] // 2

        np_img = np_img[locx - self.patchSize[0] // 2:locx + self.patchSize[0] // 2,
                 locy - self.patchSize[1] // 2: locy + self.patchSize[1] // 2]
        np_mask = np_mask[locx - self.patchSize[0] // 2:locx + self.patchSize[0] // 2,
                  locy - self.patchSize[1] // 2: locy + self.patchSize[1] // 2]

        np_img = np_img.astype(np.float)
        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, axis=2)
        np_img -= (71.77478273, 42.71332587, 21.5973989)
        np_img /= (48.09435639, 28.02723564, 16.38324569)
        # HWC to CHW

        np_img = np_img.transpose((2, 0, 1))
        np_mask = np.squeeze(np_mask)
        return np_img, np_mask

    def __getitem__(self, i):
        idx = self.ids[i]

        img_file = osp.join(self.imgs_dir, idx + ".jpg")
        mask_file = osp.join(self.masks_dir, idx + ".bmp")

        assert osp.exists(img_file), f'img file not exist: {img_file}'

        assert osp.exists(mask_file), f'mask file not exist: {mask_file}'

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocessAll(img, mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


def _gen_test():
    from torch.utils.data import DataLoader
    import PIL.Image as Image
    pd = PatchDataset(r"D:\data\huaxiSkin\hands_opt\JPEGImages", r"D:\data\huaxiSkin\hands_opt\SegmentationPIL")
    dl = DataLoader(pd, 1)
    output_dir = r"D:\data\huaxiSkin\hands_opt\genTest"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    for idx, infos in enumerate(dl):
        img = infos['image']
        mask = infos['mask']
        npimg = img.numpy()[0]
        npmask = mask.numpy()[0]
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg *= (48.09435639, 28.02723564, 16.38324569)
        npimg += (71.77478273, 42.71332587, 21.5973989)
        npmask *= 120

        npimg = np.array(npimg, dtype=np.uint8)
        pilImg = Image.fromarray(npimg)
        pilMask = Image.fromarray(npmask)

        pilImg.save(osp.join(output_dir, str(idx) + ".jpg"))
        pilMask.save(osp.join(output_dir, str(idx) + ".bmp"))


if __name__ == '__main__':
    _gen_test()
