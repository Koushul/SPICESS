import torch.nn.functional as F
import torch.nn as  nn

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=F.leaky_relu, batchnorm=False,bias=False):
        super(LinearBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.batchnorm = batchnorm
        self.linearlayer = nn.Linear(input_dim,output_dim,bias=bias)
        if batchnorm:
            self.batchnormlayer = nn.BatchNorm1d(output_dim)
    
    def forward(self,inputx):
        inputx = F.dropout(inputx, self.dropout, self.training)
        output = self.linearlayer(inputx)
        if self.batchnorm:
            output = self.batchnormlayer(output)
        output = self.act(output)
        return output
    



