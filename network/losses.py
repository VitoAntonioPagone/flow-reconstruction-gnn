import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:,:,1:,:])
        count_w = self.tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class MaskedMSELoss(torch.nn.Module):
    """ Custom pixel-wise MSE loss"""
    def __init__(self, device):
        super(MaskedMSELoss, self).__init__()
        self.device = device

    def forward(self, predicted, target, mask):
        predicted = predicted.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)  # Mask is not used for the loss calculation

        # Calculate the difference between predicted and target
        diff = predicted - target

        # Pixel-wise mean of the squared difference
        loss_value = torch.mean(diff ** 2)

        return loss_value


class NavierStokesLoss(nn.Module):
    def __init__(self, device):
        super(NavierStokesLoss, self).__init__()

        # Finite difference kernels for dx and dy (centered differences)
        self.dx_kernel = torch.Tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]]).float().to(device)
        self.dy_kernel = torch.Tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]]).float().to(device)

    def forward(self, preds):
        u, v = preds[:, 0], preds[:, 1]  # assuming u and v are the first two channels

        # Calculate gradients
        du_dx = F.conv2d(u.unsqueeze(1), self.dx_kernel, padding=1)
        du_dy = F.conv2d(u.unsqueeze(1), self.dy_kernel, padding=1)
        dv_dx = F.conv2d(v.unsqueeze(1), self.dx_kernel, padding=1)
        dv_dy = F.conv2d(v.unsqueeze(1), self.dy_kernel, padding=1)

        # Momentum equations
        momentum_u = torch.abs(u * du_dx + v * du_dy)
        momentum_v = torch.abs(u * dv_dx + v * dv_dy)

        # Continuity equation
        continuity = torch.abs(du_dx + dv_dy)

        return (continuity + momentum_u + momentum_v).mean()

import torch
import torch_scatter
from torch_geometric.utils import get_laplacian

class GraphNavierStokesLoss(torch.nn.Module):
    def __init__(self):
        super(GraphNavierStokesLoss, self).__init__()

    def forward(self, data):
        u, v, p, nu = data.x[:, 0], data.x[:, 1], data.x[:, 4], data.x[:, 5]
        x, y = data.x[:, 6], data.x[:, 7]

        # Get the Laplacian matrix in a sparse format
        laplacian_indices, laplacian_values = get_laplacian(data.edge_index, normalization=None)
        num_nodes = data.num_nodes
        laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))

        # Compute gradients using edge relations
        du_dx = (u[data.edge_index[1]] - u[data.edge_index[0]]) / (x[data.edge_index[1]] - x[data.edge_index[0]])
        du_dy = (u[data.edge_index[1]] - u[data.edge_index[0]]) / (y[data.edge_index[1]] - y[data.edge_index[0]])
        dv_dx = (v[data.edge_index[1]] - v[data.edge_index[0]]) / (x[data.edge_index[1]] - x[data.edge_index[0]])
        dv_dy = (v[data.edge_index[1]] - v[data.edge_index[0]]) / (y[data.edge_index[1]] - y[data.edge_index[0]])

        # Continuity Loss (based on divergence-free condition)
        div_u = du_dx + dv_dy
        continuity_loss = div_u.abs().mean()

        # Convection terms
        conv_u = u * du_dx + v * du_dy
        conv_v = u * dv_dx + v * dv_dy

        # Second order viscous terms (Laplacian)
        # Using the sparse Laplacian matrix for computing the second order terms
        laplacian_u = torch.sparse.mm(laplacian, u.unsqueeze(-1)).squeeze()
        laplacian_v = torch.sparse.mm(laplacian, v.unsqueeze(-1)).squeeze()

        # Pressure gradients
        dp_dx = (p[data.edge_index[1]] - p[data.edge_index[0]]) / (x[data.edge_index[1]] - x[data.edge_index[0]])
        dp_dy = (p[data.edge_index[1]] - p[data.edge_index[0]]) / (y[data.edge_index[1]] - y[data.edge_index[0]])

        # Momentum Loss for u and v components
        momentum_loss_u = (conv_u + dp_dx - nu * laplacian_u).abs().mean()
        momentum_loss_v = (conv_v + dp_dy - nu * laplacian_v).abs().mean()

        total_momentum_loss = momentum_loss_u + momentum_loss_v

        # Return the sum of the continuity loss and the total momentum loss
        return continuity_loss + total_momentum_loss







