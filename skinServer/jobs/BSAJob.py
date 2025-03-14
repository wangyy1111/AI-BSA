import os
from .baseJob import BaseJob, errorInfo
import json
from models.handSegModel import HandSegModel
from models.handSegOnnxModel import HandSegOnnxModel
from models.nidusSegModel import NidusSegModel
import base64
import PIL.Image as Image
import tempfile
import uuid
from utils.io import file2b64
import numpy as np

from utils.errorCode import errorCode, getErrorInfo

from skimage import transform

import logging

tempPath = r"../../temp"
logPath = r"../../log"

if not os.path.exists(logPath):
    os.makedirs(logPath)


def string_to_file(string):
    file_like_obj = tempfile.NamedTemporaryFile()
    file_like_obj.write(string)
    file_like_obj.flush()
    file_like_obj.seek(0)
    return file_like_obj


def pil_to_string(pilImage):
    # fp = tempfile.NamedTemporaryFile()
    uid = str(uuid.uuid4())
    suid = ''.join(uid.split('-'))
    pilImage.save(os.path.join(tempPath, suid + '.png'))

    return file2b64(os.path.join(tempPath, suid + '.png'))


useOnnx = False


class BSAJob(BaseJob):
    def __init__(self):
        super(BSAJob, self).__init__()
        if useOnnx:
            self.handSegModel = HandSegOnnxModel()
        else:
            self.handSegModel = HandSegModel()
        print("init the models.")

        self.nidusSegModel = NidusSegModel()

        self.logger = logging.getLogger("bsaLogger")
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(logPath, "log.txt"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def run(self, request):
        """"""
        # return self._tjob(request)
        return self._dojob(request)

    def _try_get(self, d, key, dv=None):
        if key not in d:
            return dv
        return d[key]

    def _dojob(self, request):
        if isinstance(request.json, str):
            req = json.loads(request.json)
        else:
            req = request.json
        imgs = self._try_get(req, "list")
        if not imgs:
            pass  # worry key.
        if not isinstance(imgs, list):
            pass  # not a list.

        #  params init.
        handFind = 0

        retInfo = {}
        retInfo["code"] = errorCode.success
        retInfo["message"] = getErrorInfo(errorCode.success)
        retInfo["palm"] = []
        retInfo["segments"] = []
        try:
            for img in imgs:
                if img["palmtype"] == "1" and handFind == 0:
                    handFind = 1
                    handimg = img["imgdata"]
                    fp = string_to_file(base64.b64decode(bytes(handimg, encoding="utf-8")))
                    area, pred, suc = self.handSegModel.run(fp)

                    print(pred)
                    print(pred.shape)

                    self.logger.info(f"hand segment with area result :{area}")

                    # ############
                    palmpixel = np.sum(pred != 0)
                    predsig = pred.astype(np.uint8)
                    predsig[pred == 2] = 1
                    predsig[pred != 2] = 0
                    predhand = pred.astype(np.uint8)
                    predhand[pred == 1] = 255
                    predhand[pred != 1] = 0
                    #print(predsig.shape)
                    #print(predhand.shape)
                    # predsig = transform.resize(predsig, predhand.shape, preserve_range=True)
                    predhand = np.stack([(predsig * 250).astype(np.uint8), (predhand * 0.2).astype(np.uint8), predhand,
                                    (predhand / 4).astype(np.uint8) + (predsig * 120).astype(np.uint8)],
                                   -1)  # to png with alpha.
                    #print(predhand.shape)
                    piltImg = Image.fromarray(predhand)
                    handimg = pil_to_string(piltImg)
                    #print(handimg)
                    #print(type(handimg))
                    retInfo["code"] = suc
                    retInfo["palm"].append({"id": img["id"], "palmarea": area, "palmpixel":int(palmpixel), "handsegdata": handimg})  # , "handsegdata": ,
                    
                elif img["palmtype"] == 1 and handFind == 1:
                    retInfo["code"] = errorCode.mh

                elif img["palmtype"] == "2":
                    fp = string_to_file(base64.b64decode(bytes(img["imgdata"], encoding="utf-8")))
                    seg, signPred, area, suc, segpixel  = self.nidusSegModel.run(fp)
                    self.logger.info(f"Nidus segment with area result :{area}")
                    signPred = transform.resize(signPred, seg.shape, preserve_range=True)
                    seg = np.stack([(signPred * 250).astype(np.uint8), (seg * 0.2).astype(np.uint8), seg,
                                    (seg / 4).astype(np.uint8) + (signPred * 120).astype(np.uint8)],
                                   -1)  # to png with alpha.
                    pilImg = Image.fromarray(seg)
                    simg = pil_to_string(pilImg)
                    retInfo["code"] = suc
                    retInfo["segments"].append({"segdata": simg, "segarea": area, "id": img["id"], "segpixel":int(segpixel)})

                if retInfo["code"] != errorCode.success:
                    retInfo["message"] = getErrorInfo(retInfo["code"])
                    self.logger.warning(f"break with info :{getErrorInfo(retInfo['code'])}")
                    break

        except Exception as e:
            self.logger.warning(f"Error with :{e}")
            retInfo["code"] = errorCode.unknown

        # print(retInfo)
        return retInfo

    def _tjob(self, request):
        """only used for the """
        # print(request)
        # req = json.loads(request.json)
        print(type(request.json))
        if isinstance(request.json, str):
            req = json.loads(request.json)
        else:
            req = request.json
        print(req)
        imgs = req["list"]
        segimg = []
        segarea = []
        for img in imgs:
            if img["palmtype"] == "1":
                handimg = img["imgdata"]
                with open("../../", "wb") as f:
                    f.write(base64.b64decode(bytes(handimg, encoding="utf-8")))
            else:
                segimg.append(img["imgdata"])
                segarea.append(1.2)
        res = {}
        res["status"] = 0
        res["palmarea"] = 1.23
        res["segimg"] = imgs
        res["segarea"] = segarea
        return json.dumps(res)
