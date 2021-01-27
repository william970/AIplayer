class TransformerConfig(object):
    def __init__(
            self,
            d_model=1000,
            n_layers=12,
            heads=10,
            dropout=0.1
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
