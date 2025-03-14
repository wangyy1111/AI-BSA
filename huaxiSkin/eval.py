import torch
import torch.nn.functional as F
from tqdm import tqdm


# from dice_loss import dice_coeff,DiceLoss


def eval_net(net, loader, device, output_chanels):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    #     mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    tot1 = 0
    tot2 = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if output_chanels > 1:
                mask_pred = F.softmax(mask_pred, dim=1)
                mask_pred = torch.argmax(mask_pred, dim=1)
                mask_pred = mask_pred.float()

                #                 print(mask_pred.max())

                #                 tot += dice_coeff(mask_pred, true_masks).item()

                # 分别计算了视杯（cup）和视盘（disc）的预测结果和标签
                one = torch.ones_like(true_masks)
                zero = torch.zeros_like(true_masks)
                disc_pred = torch.where(mask_pred == 1, one, zero)
                cup_pred = torch.where(mask_pred == 2, one, zero)
                disc_mask = torch.where(true_masks == 1, one, zero)
                cup_mask = torch.where(true_masks == 2, one, zero)

                intersection = disc_pred.view(-1) * disc_mask.view(-1)
                disc_dice = 2 * intersection.sum() / (disc_pred.view(-1).sum() + disc_mask.view(-1).sum() + 1e-5)

                intersection = cup_pred.view(-1) * cup_mask.view(-1)
                cup_dice = 2 * intersection.sum() / (cup_pred.view(-1).sum() + cup_mask.view(-1).sum() + 1e-5)

                tot1 += disc_dice.item()
                tot2 += cup_dice.item()

            #                 tot1 +=(1-DiceLoss()(disc_pred, disc_mask)).item()
            #                 tot2 +=(1-DiceLoss()(cup_pred, cup_mask)).item()
            #                 print(disc_pred.shape)
            #                 print(cup_pred.shape)
            #                 tot1 +=(1-DiceLoss()(mask_pred[mask_pred==1], true_masks[true_masks==1])).item()
            #                 tot2 +=(1-DiceLoss()(mask_pred[mask_pred==2], true_masks[true_masks==2])).item()

            #                 tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                # tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot1 / n_val, tot2 / n_val
