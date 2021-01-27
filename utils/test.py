import torch
import torch.nn.functional as F
import numpy as np

# 操作序列 = np.ones((1, ))
# 操作序列[0]=128
# print(操作序列.shape)
# input = torch.randn(2, 3, 4)
# print(input)
# b = F.softmax(input, dim=-1)  # 按列SoftMax,列和为1
# print(b)
# print(b.shape)
# b = b[:, - 1, :]
# print(b)
# print(b.shape)
# b = torch.multinomial(b, num_samples=1)
# print(b)
# print(b.shape)
# print(b[0][0])
# b = b.cpu().numpy()
# print(b)
# print(b.shape)
# print(b[0, 0])
# 操作序列 = np.append(操作序列, b[0, 0])
# print(操作序列)
# print(操作序列.shape)
# 操作序列 = np.append(操作序列,b[0, 0])
# print(操作序列)
# print(操作序列.shape)
# 操作序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda()
# print(操作序列.shape)
# 操作序列 = 操作序列.unsqueeze(0)
# print(操作序列.shape)
# from sklearn.datasets import load_digits
# from sklearn.manifold import MDS
# X, _ = load_digits(return_X_y=True)
# print(X[0])
#
# embedding = MDS(n_components=100)
# X_transformed = embedding.fit_transform(X[:2])
# print(X_transformed[0])
# embedding2 = MDS(n_components=64)
# X_transformed2 = embedding2.fit_transform(X_transformed)
# print(X_transformed2[0])
# opScr = [123]
# opScr2 = [456, 57]
# opScr = opScr.extend(opScr2)
# print(opScr)
import copy

#
#
reslist = genOnehot(8)
print(len(reslist))
