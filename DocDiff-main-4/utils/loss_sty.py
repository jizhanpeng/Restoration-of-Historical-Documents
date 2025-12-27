import torch
import torch.nn as nn
import torch.nn.functional as F


# feat_output_comp 需要进行一个替换，只保留受损区域，其他区域用真值换
class CPLoss(nn.Module):
    def __init__(self):
        super(CPLoss, self).__init__()

    def forward(self, output_comp, gt):
        prc_loss = self.percetual_loss(output_comp, gt)

        losses = prc_loss

        return losses

    def percetual_loss(self, feat_output_comp, feat_gt):
        pcr_losses = []
        for i in range(3):
            pcr_losses.append(F.l1_loss(feat_output_comp[i], feat_gt[i]))
        return sum(pcr_losses)