from .baseModel import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import PIL.Image as Image

import numpy as np

modelPath = '../../'


class HandSegModel(BaseModel):
    def __init__(self, imgSize=(512, 512), baseSize=7.5 * 7):  # baseSize in mm^2.
        super().__init__()
        self.net = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=3, decoder_attention_type='scse')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(modelPath, map_location=self.device))
        self.net.eval()

        self.imgSize = imgSize
        self.baseSize = baseSize

    def _calcHandArea(self, nparr: np.ndarray):
        print(np.sum(nparr == 1), np.sum(nparr == 2))
        if np.sum(nparr == 2) < 10:
            return -1.0
        return np.sum(nparr == 1) / np.sum(nparr == 2) * self.baseSize

    def _predImg(self, imgPath):
        pred = self.net.forward(self._transform_image(imgPath))
        pred = pred.squeeze().cpu().detach().numpy()
        return np.argmax(pred, axis=0)

    def _transform_image(self, imgPath):
        full_img = Image.open(imgPath)
        pil_img = full_img.resize((self.imgSize[0], self.imgSize[1]))
        npimg = np.array(pil_img)
        npimg = npimg.astype(np.float32)
        npimg -= (71.77478273, 42.71332587, 21.5973989)
        npimg /= (48.09435639, 28.02723564, 16.38324569)
        npimg = npimg.transpose((2, 0, 1))

        img = torch.from_numpy(npimg)
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        return img

    def _transform_image_pt(self, imgPath):
        """ not use now. """
        my_transforms = transforms.Compose([transforms.Resize(512),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(imgPath)
        return my_transforms(image).unsqueeze(0)

    def _save4test(self, img):
        import PIL.Image as Image
        print(img.shape)
        img = np.array(img, dtype=np.uint8)
        img *= 120
        plimg = Image.fromarray(img)
        plimg.save("../../")

    def run(self, img):
        pred = self._predImg(img)
        # self._save4test(pred)
        return self._calcHandArea(pred), pred
