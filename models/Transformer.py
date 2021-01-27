import torch
import torch.nn as nn
from models.Layers import EncoderLayer, DecoderLayer
from models.Embed import PositionalEncoder
from models.Sublayers import Norm
from sklearn.manifold import MDS
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N

        # 词向量表
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        # x = self.embed(src)
        x = src
        z = self.pe(x)
        for i in range(self.N):
            z = self.layers[i](z, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # 词向量表
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=10, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        # x = self.embed(trg)
        x = trg
        z = self.pe(x)
        for i in range(self.N):
            z = self.layers[i](z, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)
        # self.out = nn.Linear(d_model, outputVecLength)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")

        # print(src.shape)
        # print(trg.shape)
        # X_transformed = embedding.fit_transform(X[:100])
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)

        # output = self.out(d_output)
        return d_output


def get_model(d_model, n_layers, heads, dropout):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(d_model, n_layers, heads, dropout)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if opt.device == 0:
        model = model.cuda()

    return model
