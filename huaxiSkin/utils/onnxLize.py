import os
import numpy as np
import onnx
import torch
import torch.nn as nn

import onnxruntime
import segmentation_models_pytorch as smp


def saveModel(model, path, ipt):
    """save the model to the path in onnx fromat."""
    # Export the model
    torch.onnx.export(model,  # model being run
                      ipt,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})


def checkModel(modelPath):
    onnx_model = onnx.load(modelPath)
    onnx.checker.check_model(onnx_model)


def checkResult(torchModel, onnxModelPath, ipt):
    ort_session = onnxruntime.InferenceSession(onnxModelPath)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(ipt)}
    ort_outs = ort_session.run(None, ort_inputs)

    torch_out = torchModel(ipt)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def optHandModels():
    net = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=3, decoder_attention_type='scse')
    net = nn.DataParallel(net, device_ids=[0, 1])

    model = '/root/models/huaxiSkin/0702_smpresNextUnet/checkpoints/CP_epoch99.pth'  # now best 0623 CP_epoch99.pth

    optPath = "/root/datas/REFUGE/opt_test623_smpresNextUnet/"

    optOnnxPath = ""

    if not os.path.exists(optPath):
        os.makedirs(optPath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    _, net = next(net.modules())

    iptTensor = torch.randn(1, 3, 512, 512, requires_grad=True)

    saveModel(net, optOnnxPath, iptTensor)

    checkModel(optOnnxPath)

    checkResult(net, optOnnxPath, iptTensor)

if __name__ == '__main__':
    optHandModels()
