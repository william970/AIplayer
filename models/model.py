import copy
import json

import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.transforms import Resize, Normalize
import torch.nn.functional as F
import torch
from torch import optim
import time

from models.Batch import create_masks
from models.Transformer import Transformer
from config.Config import TransformerConfig


def getIndex(valueName, dictName):
    return list(dictName.keys())[list(dictName.values()).index(valueName)]


# , trg_vocab, d_model, N, heads, dropout, 图向量尺寸=1000

class RTNet(nn.Module):
    def __init__(self, d_model, n_layers, heads, dropout, GraphVectorLength=1000):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True).eval().requires_grad_(True)
        # 保存前几帧计算的图向量
        self.GraphVectorList = Tensor().cuda()
        self.opList = Tensor().cuda()
        self.FullConnection = nn.Linear(512 * 26 * 15, GraphVectorLength)
        self.transformer = Transformer(d_model, n_layers, heads, dropout)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        # self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, img, op):
        img = img.unsqueeze(0)
        print(img.shape)
        # print(img.type())
        # print(type(img))
        if self.GraphVectorList.shape[0] < 10:
            with torch.no_grad():
                output = self.resnet.conv1(img)
                output = self.resnet.bn1(output)
                output = self.resnet.relu(output)
                output = self.resnet.maxpool(output)

                output = self.resnet.layer1(output)
                output = self.resnet.layer2(output)
                output = self.resnet.layer3(output)
                output = self.resnet.layer4(output)
                # print(output.shape)
                output = F.adaptive_avg_pool2d(output, [26, 15]).permute(0, 2, 3, 1)
                # print(output.shape)
                output = output.reshape(output.shape[0], output.shape[1] * output.shape[2] * output.shape[3])
                output = self.FullConnection(output)
                self.GraphVectorList = torch.cat((self.GraphVectorList, output))
                self.opList = torch.cat((self.opList, op))
                return None
        else:
            # self.GraphVectorList[0] = self.GraphVectorList[0].cpu()
            temp = self.GraphVectorList[1:]
            self.GraphVectorList = []
            self.GraphVectorList = temp
            tep = self.opList[1:]
            self.opList = []
            self.opList = tep
            output = self.resnet.conv1(img)
            output = self.resnet.bn1(output)
            output = self.resnet.relu(output)
            output = self.resnet.maxpool(output)

            output = self.resnet.layer1(output)
            output = self.resnet.layer2(output)
            output = self.resnet.layer3(output)
            output = self.resnet.layer4(output)
            output = F.adaptive_avg_pool2d(output, [26, 15]).permute(0, 2, 3, 1)
            output = output.reshape(output.shape[0], output.shape[1] * output.shape[2] * output.shape[3])
            output = self.FullConnection(output)
            # print(self.GraphVectorList.shape)
            self.GraphVectorList = torch.cat((self.GraphVectorList, output))
            self.opList = torch.cat((self.opList, op))
            print(self.opList.shape)
            print(self.GraphVectorList.shape)
            tgt = torch.Tensor(10).unsqueeze(0)
            src = torch.Tensor(10).unsqueeze(0)
            src_mask, trg_mask = create_masks(src, tgt)
            # print(self.GraphVectorList.shape)
            # print(self.GraphVectorList.shape)
            # print(src_mask)
            # print(trg_mask)

            output2 = self.transformer(self.GraphVectorList, self.opList, src_mask, trg_mask)

            # d_output = self.decoder(图向量, trg_mask)
            # output = self.out(d_output)
            # output = output[:, -1, :]
            return output2

# config = TransformerConfig()
# d_model = config.d_model
# n_layers = config.n_layers
# heads = config.heads
# dropout = config.dropout
# resnet50 = RTNet(d_model, n_layers, heads, dropout)
# resnet50 = resnet50.cuda()
# opt = optim.SGD(resnet50.parameters(), lr=0.01)
# f = open('../config/embedding.txt', 'r')
# op_embedding_str = f.read()
# op_embedding_Dict = json.loads(op_embedding_str)
# loss_func = F.cross_entropy
# f.close()
#
# f = open('../config/opDict.txt', 'r')
# op_str = f.read()
# op_Dict = json.loads(op_str)
# f.close()
#
# loss_func = torch.nn.MSELoss(reduction='mean')
#
# for i in range(20):
#     x = torch.randn(1, 3, 1642, 936).cuda()
#     y = list(torch.randint(0, 2, [8]))
#     # print(time.process_time())
#     index = getIndex(y, op_Dict)
#     # print(time.process_time())
#     op_embedding = op_embedding_Dict[index]
#     # print(len(op_embedding))
#     op_embedding_1 = torch.Tensor(op_embedding)
#     op_embedding_2 = op_embedding_1.reshape(1, 1000)
#     op_embedding_3 = op_embedding_2.cuda()
#     # print(op_embedding.shape)
#     out = resnet50(x, op_embedding_3)
#     if out is not None:
#         out = out.squeeze(0)
#         # print(resnet50.opList.view(-1).shape)
#         # print(resnet50.opList)
#         acc = resnet50.opList
#         print(out.view(-1, out.size(-1)).shape)
#         print(acc.view(-1, acc.size(-1)).shape)
#         loss = loss_func(out.view(-1, out.size(-1)), acc.view(-1, acc.size(-1)))
#         loss.backward()
#         print('success')
#         opt.step()
#         opt.zero_grad()
# print(resnet50.opList.shape)
