## Adapted from https://github.com/zhouzilu/cTPnet/blob/master/extdata/training_05152020.py
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr


class CTPNet(nn.Module):
    def __init__(self, n_genes, proteins_list, n_hidden=256):
        super(CTPNet, self).__init__()
        self.proteins = proteins_list
        self.fc1 = nn.Linear(n_genes, 512)
        self.fc2 = nn.Linear(512, n_hidden)

        self.fc3 = nn.ModuleDict({})

        for p in self.proteins:
            self.fc3[p] = nn.Linear(n_hidden, 64)
        
        self.fc4 = nn.ModuleDict({})
        
        for p in self.proteins:
            self.fc4[p] = nn.Linear(64, 1)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs={}
        
        for p in self.proteins:
            outputs[p] = self.fc4[p](F.relu(self.fc3[p](x)))
            
        return outputs
    
    
    @torch.no_grad()
    def test(self, Xt, yt):
        self.eval()
        outputs = self.forward(Xt)
        preds = np.column_stack([outputs[p].data.cpu().numpy() for p in self.proteins])
        corrsx = []

        yt = yt.detach().cpu().numpy()
        
        # for ixs in range(len(self.proteins)):
        #     corrsx.append(spearmanr(yt[:, ixs], preds[:, ixs]).statistic)
        
        for ixs in range(Xt.shape[0]):
            corrsx.append(spearmanr(yt[ixs], preds[ixs]).statistic)
            
        return corrsx
            
    
    def fit(self, X, y, epochs=200, lr=0.001, n_batches = 32):
        X_train = X[:]
        y_train = y[:]
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, amsgrad=True, weight_decay=0.001)
        max_epochs = epochs
        protein_list = list(self.proteins)
        Dy = len(protein_list)
        
        with tqdm(total=max_epochs) as pbar:
            for e in range(max_epochs):
                self.train()
                
                for i in range(int(y_train.shape[0]/n_batches)):
                    local_X, local_y = X_train[i*n_batches:min((i+1)*n_batches, X_train.shape[0]-1),], y_train[i*n_batches:min((i+1)*n_batches, y_train.shape[0]-1),]
                    optimizer.zero_grad() # zero the parameter gradients
                    outputs_dict = self.forward(local_X)
                    loss = None
                    loss_count = 0.0

                    for p in protein_list:
                        notNaN = (local_y[:,protein_list.index(p):(protein_list.index(p)+1)]==local_y[:,protein_list.index(p):(protein_list.index(p)+1)])
                        loss_p = criterion(outputs_dict[p][notNaN],local_y[:,protein_list.index(p):(protein_list.index(p)+1)][notNaN])

                        if not torch.isnan(loss_p):
                            loss_count += 1.0
                            if loss is None:
                                loss = loss_p
                            else:
                                loss = loss+loss_p

                    if loss is not None:
                        loss.backward()
                        optimizer.step()     
                        
                corrsx = self.test(Xt=X[:], yt=y[:])
                
                pbar.update()
                pbar.set_description(f'CORR: {np.mean(corrsx):.3f}')
                    

        return self.test(Xt=X[:], yt=y[:])