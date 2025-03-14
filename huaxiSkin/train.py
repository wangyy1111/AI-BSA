import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
# from unet import UNet
# from resunet.resunet import resNextUnet50, resUnet18, resUnet32, resUnet50
# from unet import unetpp
import segmentation_models_pytorch as smp

from torch.utils.tensorboard import SummaryWriter
# from utils.dataset import BasicDataset
from utils.patchDataset import PatchDataset
from torch.utils.data import DataLoader, random_split
from pytorch_toolbelt import losses as L

# from torchcontrib.optim import SWA

dir_img = '..'
dir_mask = '..'
dir_checkpoint = '..'

random.seed(821)
np.random.seed(821)
torch.manual_seed(821)


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=(512, 512),
              output_channel=3):
    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset = PatchDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    #     epoch_step = n_train // batch_size
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    #     opt = SWA(optimizer, swa_start=5 * epoch_step, swa_freq=epoch_step, swa_lr=lr / 10)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=4)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
    if True:  # net.n_classes > 1   or  next(net.children()).n_classes > 1
        #         criterion = L.DiceLoss("multiclass", [1,])
        criterion = nn.CrossEntropyLoss()
        # criterion = L.FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)

                mask_type = torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:  # val 1 times a epoch.
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    #                     opt.swap_swa_sgd()
                    #                     opt.bn_update(train_loader, net, device='cuda')
                    total1, total2 = eval_net(net, val_loader, device, output_channel)
                    scheduler.step()

                    print("now lr = ", optimizer.param_groups[0]['lr'])
                    print('Val hand Dice Coeff: {}'.format(total1))
                    print('Val base Dice Coeff: {}'.format(total2))


        if save_cp:
            try:
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            #             opt.swap_swa_sgd()
            #             opt.bn_update(train_loader, net, device='cuda')
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,  # 24
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        default=False,  
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=tuple, default=(512, 512),
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    #     net = UNet(n_channels=3, n_classes=3, bilinear=True)
    #     net = resUnet18()
    #     net = resUnet32()
    #     net = unetpp.NestedUNet(3)
    net = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=3, decoder_attention_type='scse')
    #     net = resUnet50()
    net = nn.DataParallel(net, device_ids=[0, 1])
    #     logging.info(f'Network:\n'
    #                  f'\t{net.modules[0].n_channels} input channels\n'
    #                  f'\t{net.modules[0].n_classes} output channels (classes)\n'
    #                  f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
