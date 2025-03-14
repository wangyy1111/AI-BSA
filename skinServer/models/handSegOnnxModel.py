
import onnxruntime
from .baseModel import BaseModel
import PIL.Image as Image
import numpy as np

onnxWeightPath = r"../.."


class HandSegOnnxModel(BaseModel):
    def __init__(self, imgSize=(512, 512), baseSize=7.5 * 7):
        super(HandSegOnnxModel, self).__init__()

        self.imgSize = imgSize
        self.baseSize = baseSize

        self.infer = onnxruntime.InferenceSession(onnxWeightPath)

    def _transform_image(self, imgPath):
        full_img = Image.open(imgPath)
        pil_img = full_img.resize((self.imgSize[0], self.imgSize[1]))
        npimg = np.array(pil_img)
        npimg = npimg.astype(np.float32)
        npimg -= (71.77478273, 42.71332587, 21.5973989)
        npimg /= (48.09435639, 28.02723564, 16.38324569)
        npimg = npimg.transpose((2, 0, 1))
        npimg = npimg[np.newaxis, :, :, :]
        return npimg

    def _calcHandArea(self, nparr: np.ndarray):
        if np.sum(nparr == 2) < 10:
            return -1.0
        return np.sum(nparr == 1) / np.sum(nparr == 2) * self.baseSize

    def _predImg(self, imgPath):
        print("hangSegOnnx predicting")
        pred = self.infer.run(None, {self.infer.get_inputs()[0].name: self._transform_image(imgPath)})
        print("after hangSegOnnx predicting.", pred[0].shape)
        return np.argmax(pred[0][0], axis=0)

    def run(self, request):
        pred = self._predImg(request)
        return self._calcHandArea(pred), pred


if __name__ == '__main__':
    pass
