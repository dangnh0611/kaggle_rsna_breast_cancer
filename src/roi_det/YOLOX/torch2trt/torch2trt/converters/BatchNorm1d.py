from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.BatchNorm1d.forward')
def convert_BatchNorm1d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    scale = module.weight.detach().cpu().numpy() / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
    bias = module.bias.detach().cpu().numpy() - module.running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)
    
    # reshape to 2D
    layer = ctx.network.add_shuffle(input_trt)
    
    if len(input.shape) == 2:
        layer.reshape_dims = (0, 0, 1, 1)
    else:
        layer.reshape_dims = (0, 0, 0, 1)
    
    layer = ctx.network.add_scale(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)

    # reshape back to 1D
    layer = ctx.network.add_shuffle(layer.get_output(0))
    if len(input.shape) == 2:
        layer.reshape_dims = (0, 0)
    else:
        layer.reshape_dims = (0, 0, 0)
    
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 10, 3)], max_batch_size=2)
def test_BatchNorm1d_basic():
    return torch.nn.BatchNorm1d(10)