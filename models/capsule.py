"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3, **kwargs):
        super(CapsuleLayer, self).__init__()

        self.opt = kwargs['opt']
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            x = x.transpose(1, 2).contiguous()
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = torch.zeros(*priors.size()).to(torch.device("cuda" if self.opt.cuda else "cpu"))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

            outputs = outputs.squeeze()
            if outputs.ndimension() == 2:
                outputs = outputs.unsqueeze(1)
            outputs = outputs.transpose(0, 1).contiguous()
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        outputs = outputs.transpose(1, 2).contiguous()
        # print(outputs.shape)
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CapsuleNet, self).__init__()

        self.opt = kwargs['opt']
        self.features = self._make_features(cfg[1], cfg[0])
        self.capsules = self._make_capsules(cfg[2])
        self.decoder = self._make_decoder(cfg[4], cfg[3])

    def forward(self, x, y=None):
        x = self.features(x)
        print(x.shape)
        x = self.capsules(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x

    def _make_features(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'B':
                layers += [nn.BatchNorm2d(in_channels)]
            elif isinstance(v, str):
                v = v.split('_')
                layers += [nn.Conv2d(in_channels, int(v[0]), kernel_size=int(v[1]), stride=int(v[2]), padding=int(v[3]))]
                for i in range(0, len(v) - 4):
                    if v[i + 4] == 'B':
                        layers += [nn.BatchNorm2d(int(v[0]))]
                    elif v[i + 4] == 'R':
                        layers += [nn.ReLU(inplace=True)]
                in_channels = int(v[0])
        return nn.Sequential(*layers)

    def _make_capsules(self, cfg):
        layers = []
        for v in cfg:
            if isinstance(v, str):
                v = v.split('_')
                if len(v) == 6:
                    layers += [CapsuleLayer(num_capsules=int(v[0]), num_route_nodes=int(v[1]), in_channels=int(v[2]),
                                            out_channels=int(v[3]), kernel_size=int(v[4]), stride=int(v[5]), opt=self.opt)]
                elif len(v) == 4:
                    layers += [CapsuleLayer(num_capsules=int(v[0]), num_route_nodes=int(v[1]), in_channels=int(v[2]),
                                            out_channels=int(v[3]), opt=self.opt)]
                elif len(v) == 7:
                    layers += [CapsuleLayer(num_capsules=int(v[0]), num_route_nodes=int(v[1]), in_channels=int(v[2]),
                                            out_channels=int(v[3]), kernel_size=int(v[4]), stride=int(v[5]),
                                            opt=self.opt),
                               nn.BatchNorm1d(int(v[6]))]
                elif len(v) == 5:
                    layers += [CapsuleLayer(num_capsules=int(v[0]), num_route_nodes=int(v[1]), in_channels=int(v[2]),
                                            out_channels=int(v[3]), opt=self.opt),
                               nn.BatchNorm1d(int(v[4]))]
        return nn.Sequential(*layers)

    def _make_decoder(self, cfg, in_features):
        layers = []
        for v in cfg:
            if isinstance(v, str):
                v = v.split('_')
                layers += [nn.Linear(in_features, int(v[0]))]
                for i in range(0, len(v) - 1):
                    if v[i + 1] == 'R':
                        layers += [nn.ReLU(inplace=True)]
                    elif v[i + 1] == 'D':
                        layers += [nn.Dropout(0.5)]
                in_features = int(v[0])
        return nn.Sequential(*layers)


cfgs = {
    '': [1, ['B', '32_9_1_0_R_B'], ['8_-1_32_4_9_2', '128_800_8_16'], 2048, ['3']],
    'w': [1, ['B', '32_9_1_0_R_B'], ['16_-1_32_8_9_2', '128_1600_16_16'], 2048, ['3']],
    '7': [1, ['B', '32_9_1_0_R_B', '64_3_1_1_R_B', '64_3_1_1_R_B', '128_3_1_1_R_B', '128_3_1_1_R_B',
              '256_3_1_1_R_B'], ['8_-1_256_4_9_2', '128_800_8_16'], 2048, ['3']],
    '12': [1, ['B', '32_9_1_0_R_B', '64_3_1_1_R_B', '64_3_1_1_R_B', '128_3_1_1_R_B', '128_3_1_1_R_B',
               '256_3_1_1_R_B', '256_3_1_1_R_B', '512_3_1_1_R_B', '512_3_1_1_R_B'],
           ['8_-1_512_4_9_2', '128_800_8_16'], 2048, ['3']],
    '3': [1, ['B', '32_9_1_0_R_B'], ['8_-1_32_4_9_2', '128_800_8_16', '128_128_16_16'], 2048, ['3']],
    '3w': [1, ['B', '32_9_1_0_R_B'], ['8_-1_32_4_9_2', '512_800_8_16', '128_512_16_16'], 2048, ['3']],
    '4': [1, ['B', '32_9_1_0_R_B'], ['8_-1_32_4_9_2', '128_800_8_16', '128_128_16_16', '128_128_16_16'],
          2048, ['3']],
    '5': [1, ['B', '32_9_1_0_R_B'], ['8_-1_32_4_9_2', '128_800_8_16', '128_128_16_16', '128_128_16_16',
                                     '128_128_16_16'], 2048, ['3']],
}


def capsule(opt, **kwargs):
    model = CapsuleNet(cfgs[opt], opt=kwargs['cfg'])
    return model


def test_capsule(cuda):
    import config

    device = torch.device("cuda" if cuda else "cpu")

    for m in list(cfgs.keys()):
        print("Testing Capsule_" + (m if m != '' else 'base') + " ...")
        cfg = eval("config.capsule" + ('_' if m != '' else '') + m + "()")
        cfg.cuda = cuda
        input = torch.randn(1, 1, 35, 55)
        print(input.size())
        input = input.to(device)
        model = eval("capsule('" + m + "', cfg=cfg).to(device)")
        output = model(input)
        print(output.size())
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)


if __name__ == "__main__":
    test_capsule(True)
