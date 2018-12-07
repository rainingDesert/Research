from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.autograd import Function
# import pdb
# pdb.set_trace()


class PreHook(Function):
    
    @staticmethod
    def forward(ctx, input, offset):
        ctx.save_for_backward(input, offset)
        return input.clone()
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        input, offset = ctx.saved_variables
        return (input - offset) * grad_outputs[-1], None
    
class PostHook(Function):
    
    @staticmethod
    def forward(ctx, input, norm_factor):
        ctx.save_for_backward(norm_factor)
        return input.clone()
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        norm_factor, = ctx.saved_variables
        eps = 1e-10
        zero_mask = norm_factor < eps
        grad_input = grad_outputs[-1] / (torch.abs(norm_factor) + eps)
        grad_input[zero_mask.detach()] = 0
        return None, grad_input


def pr_conv2d(self, input):
    offset = input.min().detach()
    input = PreHook.apply(input, offset)
    resp = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).detach()
    pos_weight = F.relu(self.weight).detach()
    norm_factor = F.conv2d(input - offset, pos_weight, None, self.stride, self.padding, self.dilation, self.groups)
    output = PostHook.apply(resp, norm_factor)
    return output
