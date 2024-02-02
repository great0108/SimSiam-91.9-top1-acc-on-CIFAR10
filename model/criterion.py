from torch import nn
import torch


class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):

        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=z_a.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

    # def forward(self, y1, y2):
    #     z1 = self.projector(self.backbone(y1))
    #     z2 = self.projector(self.backbone(y2))

    #     # empirical cross-correlation matrix
    #     c = self.bn(z1).T @ self.bn(z2)

    #     # sum the cross-correlation matrix between all gpus
    #     c.div_(self.args.batch_size)
    #     torch.distributed.all_reduce(c)

    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = off_diagonal(c).pow_(2).sum()
    #     loss = on_diag + self.args.lambd * off_diag
    #     return loss