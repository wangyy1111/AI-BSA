
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import torch.nn as nn

import segmentation_models_pytorch as smp

from utils.dice import dice_with_categorical


def predict_img(net,
                full_img,
                device,
                scale_factor=(512, 512)):
    net.eval()

    newW, newH = scale_factor[0], scale_factor[1]
    pil_img = full_img.resize((newW, newH))
    npimg = np.array(pil_img)
    npimg = npimg.astype(np.float64)
    npimg -= (71.77478273, 42.71332587, 21.5973989)
    npimg /= (48.09435639, 28.02723564, 16.38324569)
    npimg = npimg.transpose((2, 0, 1))

    img = torch.from_numpy(npimg)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if True:  # next(net.children()).n_classes > 1
            probs = F.softmax(output, dim=1)
        # else:
        #     probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1], full_img.size[0])),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

        full_mask = np.argmax(full_mask, axis=0)

        return full_mask


def mask_to_image(mask):
    return Image.fromarray((mask * 120).astype(np.uint8))


if __name__ == "__main__":

    Total_disc_dice = 0
    Total_cup_dice = 0

    scale_factor = 0.5

    tets_data_files = '..'
    label_files = '..'

    net = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=3, decoder_attention_type='scse')
    net = nn.DataParallel(net, device_ids=[0, 1])

    model = '..'  # now best 0623 CP_epoch99.pth

    optPath = ".."

    if not os.path.exists(optPath):
        os.makedirs(optPath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info("Model loaded !")

    dices = []

    for i, fn in enumerate(os.listdir(tets_data_files)):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(tets_data_files + fn)

        pred = predict_img(net=net,
                           full_img=img,
                           scale_factor=(768, 768),
                           out_threshold=0.5,
                           device=device)

        result = mask_to_image(pred)

        result.save(optPath + fn[:-4] + '.bmp')  # change to .bmp for test.

        logging.info("Mask saved to {}".format(optPath + fn))

        mask = Image.open(label_files + fn[:-4] + '.bmp')
        mask = np.array(mask)
        mask /= 120

        dice = dice_with_categorical(mask, pred)

        dices.append(dice)

        # mask = 255 - mask
    #     mask[mask == 127] = 1
    #     mask[mask == 255] = 2
    #
    #     disc_pred = (pred == 1).astype(int)
    #     cup_pred = (pred == 2).astype(int)
    #     disc_mask = (mask == 1).astype(int)
    #     cup_mask = (mask == 2).astype(int)
    #
    #     ins = disc_pred * disc_mask
    #     disc_dice = 2 * (np.sum(ins) + 1e-6) / (np.sum(disc_pred) + np.sum(disc_mask) + 1e-6)
    #
    #     ins = cup_pred * cup_mask
    #     cup_dice = 2 * (np.sum(ins) + 1e-6) / (np.sum(cup_pred) + np.sum(cup_mask) + 1e-6)
    #
    #     if disc_dice < 0.6 or cup_dice < 0.6:
    #         print(str(fn) + ' disc_dice is ' + str(disc_dice) + ', cup_dice is ' + str(cup_dice))
    #
    #     Total_disc_dice += disc_dice
    #     Total_cup_dice += cup_dice
    #
    # case_disc_dice = Total_disc_dice / (i + 1)
    # case_cup_dice = Total_cup_dice / (i + 1)
    meanDice = np.mean(dices, 1)
    print(f"case disc dice is : {meanDice[1]}, case cup dice is : {meanDice[2]}")
