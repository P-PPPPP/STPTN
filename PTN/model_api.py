import torch.nn as nn
from .model import Model

class ptn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(ptn_api, self).__init__()
        assert graph_generator is None
        self.model = Model(configs)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        predicts, loss = self.model(seq_x, seq_x_mark, seq_y_mark, epoch=args['epoch'])
        return predicts, loss