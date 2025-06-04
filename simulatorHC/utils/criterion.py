import torch
import torch.nn as nn

class Lossqt(nn.Module):

    # UPnP training loss
    def __init__(self, initial_weight_value=0.5):
        super(Lossqt, self).__init__()
        self.weightq = nn.Parameter(torch.tensor(initial_weight_value))

    def forward(self, q_pred, q_gt):

        q_sign = torch.sign(q_pred.view(-1,1,4) @ q_gt.unsqueeze(-1)).squeeze(-1)
        q_regress_errors = torch.norm(q_sign * q_pred-q_gt, dim=1).sum()
        
        loss = torch.exp(-self.weightq) * q_regress_errors + self.weightq

        return loss
    


class Lossqts(nn.Module):

    # GRPS loss
    def __init__(self, initial_weight_value=[0.5,0.5,1.0]):
        super(Lossqts, self).__init__()
        self.weightq = nn.Parameter(torch.tensor(initial_weight_value[0]))
        self.weightt = nn.Parameter(torch.tensor(initial_weight_value[1]))
        self.weights = nn.Parameter(torch.tensor(initial_weight_value[2]))

    def forward(self, q_pred, q_gt, t_pred, t_gt, s_pred, s_gt):
        s_gt = s_gt.view(-1)
        s_pred = s_pred.view(-1)

        q_sign = torch.sign(q_pred.view(-1,1,4) @ q_gt.unsqueeze(-1)).squeeze(-1)
        q_regress_errors = torch.norm(q_sign * q_pred-q_gt, dim=1).sum()

        t_regress_errors = (torch.norm(t_pred-t_gt, dim=1)/torch.norm(t_gt, dim=1, keepdim=True)).sum()

        s_regress_errors = ((s_pred-s_gt).abs()/s_gt.abs()).sum()
        
        loss = torch.exp(-self.weightq) * (q_regress_errors) + self.weightq +  \
                torch.exp(-self.weightt) * (t_regress_errors) + self.weightt + \
                torch.exp(-self.weights) * (s_regress_errors) + self.weights
        
        return loss
