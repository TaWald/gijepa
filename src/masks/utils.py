# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    D = x.size(-1)
    gathered = [torch.gather(x, 1, m.unsqueeze(-1).expand(-1, -1, D)) for m in masks]
    return torch.cat(gathered, dim=0)
