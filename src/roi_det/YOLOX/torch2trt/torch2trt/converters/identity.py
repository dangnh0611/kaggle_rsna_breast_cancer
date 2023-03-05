from torch2trt.torch2trt import *


@tensorrt_converter('torch.Tensor.contiguous')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_functional_identity(ctx):
    input = ctx.method_args[0]
    if not hasattr(input, '_trt'):
        return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    output._trt = input_trt


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_identity(ctx):
    input = ctx.method_args[1]
    if not hasattr(input, '_trt'):
        return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    output._trt = input_trt
