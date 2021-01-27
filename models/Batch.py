import torch
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    variable = Variable
    np_mask = variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg):
    src_mask = (src != -1).unsqueeze(-2)
    src_mask = src_mask.cuda()
    if trg is not None:
        trg_mask = (trg != -1).unsqueeze(-2)
        trg_mask = trg_mask.cuda()
        size = trg.size(1)  # get seq_len for matrix

        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask
        # print(trg_mask.shape)
        # print(np_mask.shape)
    else:
        trg_mask = None
    return src_mask, trg_mask

# x = torch.randn(10).cuda()
#
# src_mask, trg_mask = create_masks(x.unsqueeze(0), x.unsqueeze(0))
# print(trg_mask)
# print(src_mask)
