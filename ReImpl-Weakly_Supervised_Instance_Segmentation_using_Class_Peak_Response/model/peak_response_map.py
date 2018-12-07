import torch
import torch.nn as nn
from functions.peak_backprop import pr_conv2d
from functions.peak_stimulation import peak_stimulation
from model.resnet_50 import fc_resnet50
import torch.nn.functional as F
import numpy as np

from types import MethodType


class PeakResponseMapping(nn.Sequential):
    def __init__(self, *args, **kargs):
        super(PeakResponseMapping, self).__init__()
        for arg in args:
            super(PeakResponseMapping, self).add_module(arg._get_name(), arg)

        self.inferencing = False
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)

        self.win_size = kargs.get('win_size', [3, 5, 7])
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)

        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def forward(self, input, class_threshold=0, peak_threshold=30, retrieval_cfg=None):
        assert input.dim() == 4
        if self.inferencing:
            input.requires_grad_()

        class_response_maps = super(PeakResponseMapping, self).__getitem__(0).forward(input)
        if self.enable_peak_stimulation:
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                 mode='bilinear', align_corners=True)
            peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size,
                                                      peak_filter=self.peak_filter)

        else:
            peak_list, aggregation = None, F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        if self.inferencing:
            if not self.enable_peak_backprop:
                # extract only class-aware visual cues
                return aggregation, class_response_maps

            assert class_response_maps.size(
                0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                             peak_filter=self.peak_filter)

            peak_response_maps = []
            valid_peak_list = []
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            if len(np.where(aggregation[0].cpu().data.numpy())) > 0:
                for idx in range(peak_list.size(0)):
                    if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
                        peak_val = class_response_maps[
                            peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                        if peak_val > peak_threshold:
                            grad_output.zero_()
                            # starting from the peak
                            grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                            if input.grad is not None:
                                input.grad.zero_()
                            class_response_maps.backward(grad_output, retain_graph=True)
                            prm = input.grad.detach().sum(1).clone().clamp(min=0)
                            peak_response_maps.append(prm / prm.sum())
                            valid_peak_list.append(peak_list[idx, :])
            else:
                max_score_index = np.argmax(aggregation[0].cpu().data.numpy())
                for idx in range(peak_list.size(0)):
                    if peak_list[idx, 1] == max_score_index:
                        peak_val = class_response_maps[
                            peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                        if peak_val > peak_threshold:
                            grad_output.zero_()
                            # starting from the peak
                            grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                            if input.grad is not None:
                                input.grad.zero_()
                            class_response_maps.backward(grad_output, retain_graph=True)
                            prm = input.grad.detach().sum(1).clone().clamp(min=0)
                            peak_response_maps.append(prm / prm.sum())
                            valid_peak_list.append(peak_list[idx, :])
            # return results
            class_response_maps = class_response_maps.detach()
            aggregation = aggregation.detach()

            if len(peak_response_maps) > 0:
                valid_peak_list = torch.stack(valid_peak_list)
                peak_response_maps = torch.cat(peak_response_maps, 0)
            else:
                valid_peak_list = torch.from_numpy(np.array(valid_peak_list))
                peak_response_maps = torch.from_numpy(np.array(peak_response_maps))

            if retrieval_cfg is None:
                # classification confidence scores, class-aware and instance-aware visual cues
                return aggregation, class_response_maps, peak_list, valid_peak_list, peak_response_maps
            else:
                return None

        else:
            return aggregation

    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self


def peak_response_mapping(backbone, enable_peak_stimulation=True, enable_peak_backprop=True,
                          win_size=[3, 5, 7],
                          sub_pixel_locating_factor=1, filter_type='median'):
    model = PeakResponseMapping(backbone, enable_peak_stimulation=enable_peak_stimulation,
                                enable_peak_backprop=enable_peak_backprop, win_size=win_size,
                                sub_pixel_locating_factor=sub_pixel_locating_factor, filter_type=filter_type)

    return model


if __name__ == '__main__':
    test = peak_response_mapping(fc_resnet50(), win_size=3, sub_pixel_locating_factor=8, enable_peak_stimulation=True)
    print(test)
