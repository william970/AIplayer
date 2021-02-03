import copy
from sklearn.manifold import MDS
import json


def genOnehot(N):
    reslist = []
    for i in range(N):
        temp = copy.deepcopy(reslist)
        if not temp:
            reslist = [[0]]
            temp = [[1]]
        else:
            for member in reslist:
                member.append(0)
            for member in temp:
                member.append(1)
        reslist.extend(temp)
    return reslist


# 警告！
# 该脚本用于生成操作向量的嵌入 因为操作是一个[10,11]的向量 分别表示10帧 11个按键控制（1和0分别表示按下或者没按下）
# 采用sklearn的多维度缩放来将[10,15]转换成[10,1000] 相当于操作向量嵌入作为outputEmbedding
# 因为MDS具有随机性 所以如果不想从头训练 不要运行这个脚本
if __name__ == "__main__":
    opVec = genOnehot(11)
    opDict = {}
    for i in range(len(opVec)):
        opDict[i] = list(opVec[i])
    print(opDict)
    embedding = MDS(n_components=1000)
    opVec_embedding = embedding.fit_transform(opVec)
    op_embedding_Dict = {}
    for i in range(len(opVec_embedding)):
        op_embedding_Dict[i] = list(opVec_embedding[i])
    print(op_embedding_Dict)
    op_embedding_Dict_js = json.dumps(op_embedding_Dict, indent=1)
    opDict_js = json.dumps(opDict, indent=1)
    f = open('../config/embedding.txt', 'w')
    f.write(op_embedding_Dict_js)
    f.close()
    f = open('../config/opDict.txt', 'w')
    f.write(opDict_js)
    f.close()
