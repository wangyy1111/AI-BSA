import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def singleLize():
    net = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=3, decoder_attention_type='scse')
    net = nn.DataParallel(net, device_ids=[0, 1])

    model = '/root/models/huaxiSkin/0706_smpresNextUnet/checkpoints/CP_epoch100.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    net = net.module

    torch.save(net.state_dict(), '/root/models/huaxiSkin/0706_smpresNextUnet/checkpoints/CP_epoch100s.pth')
