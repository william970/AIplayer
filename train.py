import argparse
import json

import torch
from torch import optim

from config.Config import TransformerConfig
from models.model import RTNet, getIndex
import os
from dataSet.dataloader import GameDataSet
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data')
    parser.add_argument('--weights', type=str, default='model.pt')
    parser.add_argument('--config', type=str, default='transformer')
    parser.add_argument('--opti', type=str, default='SGD')
    parser.add_argument('--dataRoot', type=str, default='data')
    opt = parser.parse_args()
    config = None
    if opt.config == 'transformer':
        config = TransformerConfig()
    if not os.path.exists('weight'):
        os.mkdir('weight')
    d_model = config.d_model
    n_layers = config.n_layers
    heads = config.heads
    dropout = config.dropout
    rtnet = RTNet(d_model, n_layers, heads, dropout)
    rtnet = rtnet.cuda()
    opti = None
    if opt.opti == 'SGD':
        opti = optim.SGD(rtnet.parameters(), lr=0.01)

    # 读取嵌入向量
    f = open('config/embedding.txt', 'r')
    op_embedding_str = f.read()
    op_embedding_Dict = json.loads(op_embedding_str)
    f.close()

    # 读取操作向量
    f = open('config/opDict.txt', 'r')
    op_str = f.read()
    op_Dict = json.loads(op_str)
    f.close()

    # 损失函数
    # loss_func = F.cross_entropy
    loss_func = torch.nn.MSELoss(reduction='mean')

    # 数据集
    train_data = GameDataSet('./data/train.txt')
    for epoch in range(100):
        for i in range(len(train_data)):
            img, label = train_data[i]
            img = img.cuda()
            # print(img.shape)
            # print(label)
            # y = list(torch.randint(0, 2, [11]))
            # print(time.process_time())
            index = getIndex(label, op_Dict)
            # print(index)
            # print(time.process_time())
            op_embedding = op_embedding_Dict[index]
            # print(len(op_embedding))
            op_embedding_1 = torch.Tensor(op_embedding)
            op_embedding_2 = op_embedding_1.reshape(1, 1000)
            op_embedding_3 = op_embedding_2.cuda()
            # print(op_embedding.shape)
            out = rtnet(img, op_embedding_3)
            if out is not None:
                out = out.squeeze(0)
                # print(resnet50.opList.view(-1).shape)
                # print(resnet50.opList)
                acc = rtnet.opList
                # print(out.view(-1, out.size(-1)).shape)
                # print(acc.view(-1, acc.size(-1)).shape)
                loss = loss_func(out.view(-1, out.size(-1)), acc.view(-1, acc.size(-1)))
                print(loss)
                # 反向传播
                loss.backward()
                print('epoch : ' + str(epoch) + 'no : ' + str(i) + ' success')
                # 更新网络
                opti.step()
                # 去除梯度
                opti.zero_grad()
                torch.cuda.empty_cache()
        # print(rtnet.opList.shape)
        if epoch == 1:
            torch.save(rtnet, os.path.join('weight', opt.weights))
