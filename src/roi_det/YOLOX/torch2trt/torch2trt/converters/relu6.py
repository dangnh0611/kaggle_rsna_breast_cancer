from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.relu6')
def convert_functional_relu6(ctx):
    ctx.method_args = (torch.nn.ReLU6(),) + ctx.method_args
    convert_relu6(ctx)


@tensorrt_converter('torch.nn.ReLU6.forward')
def convert_relu6(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return

    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input, 6])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))

    layer = ctx.network.add_activation(
        input=input_a_trt, type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(
        layer.get_output(0), input_b_trt, trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu6_basic():
    return torch.nn.ReLU6()


class FunctionalRelu6(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu6(x)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_relu6_basic():
    return FunctionalRelu6()

