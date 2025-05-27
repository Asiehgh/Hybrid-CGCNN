from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len, eps=1e-3, momentum=0.05)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len, eps=1e-3, momentum=0.05)
        self.softplus2 = nn.Softplus()


    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """

        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, n_extra_features=None):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                             nbr_fea_len=nbr_fea_len)
                                   for _ in range(n_conv)])
        
        self.extra_feat_net = nn.Sequential(
            nn.Linear(n_extra_features, h_fea_len // 4),
            nn.BatchNorm1d(h_fea_len // 4, eps=1e-3, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_fea_len + h_fea_len // 4, h_fea_len),
            nn.BatchNorm1d(h_fea_len, eps=1e-3, momentum=0.05),
            nn.ReLU()
        )
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                    for _ in range(n_h-1)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(h_fea_len, eps=1e-3, momentum=0.05)
                                    for _ in range(n_h-1)])
        
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        
        self.dropout = nn.Dropout(0.3) 




    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea):
        atom_fea = atom_fea.float()
        nbr_fea = nbr_fea.float()
        extra_fea = extra_fea.float()

        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        extra_fea = self.extra_feat_net(extra_fea)

        combined_fea = torch.cat([crys_fea, extra_fea], dim=1)
        crys_fea = self.conv_to_fc(combined_fea)
        
        if hasattr(self, 'fcs') and hasattr(self, 'bns'):
            for fc, bn in zip(self.fcs, self.bns):
                residual = crys_fea
                crys_fea = self.dropout(bn(nn.ReLU()(fc(crys_fea))))
                crys_fea = crys_fea + residual
        
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """Keep the original pooling method unchanged"""
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
        
    def visualize_feature_processing(self, extra_fea):
        print("\nFeature Processing Visualization:")
        print(f"Input features shape: {extra_fea.shape}")
        intermediate = self.extra_feat_net[0](extra_fea)
        print(f"After first linear layer: {intermediate.shape}")
        intermediate = self.extra_feat_net[1](intermediate)
        print(f"After batch norm: {intermediate.shape}")
        intermediate = self.extra_feat_net[2](intermediate)
        print(f"After ReLU: {intermediate.shape}")
        output = self.extra_feat_net(extra_fea)
        print(f"Final extra feature output shape: {output.shape}")
        return {'input_stats': {'mean': extra_fea.mean(dim=0).tolist(),'std': extra_fea.std(dim=0).tolist()}, 'output_stats': {'mean': output.mean(dim=0).tolist(),'std': output.std(dim=0).tolist()}} 
        
        
        
        







       
              
