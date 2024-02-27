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

class SimNoiseLoss(nn.Module):
    def __init__(self, version='simplified', noise=0.1):
        super().__init__()
        self.ver = version
        self.noise = noise

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            z = z + (self.noise ** 0.5) * torch.randn(z.shape, device=z.device)
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2):

        loss1 = self.asymmetric_loss(z1, z2)
        loss2 = self.asymmetric_loss(z2, z1)

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

class BarlowTwinsLoss2(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(1, keepdim=True)) / z_a.std(1, keepdim=True) # NxD
        z_b_norm = (z_b - z_b.mean(1, keepdim=True)) / z_b.std(1, keepdim=True) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.abs(torch.mm(z_a_norm.T, z_b_norm)) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=z_a.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

   
class SimSingleLoss(nn.Module):
    def __init__(self, bias=1e-4):
        super().__init__()
        self.bias = bias

    def forward(self, z1, z2):
        z2 = z2.detach()  # stop gradient
        z1 += torch.ones(z1.shape[-1], device="cuda") * self.bias
        return - nn.functional.cosine_similarity(z1, z2, dim=-1).mean()
