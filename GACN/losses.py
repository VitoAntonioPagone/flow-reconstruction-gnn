import torch
from torch_geometric.utils import get_laplacian
from torch_scatter import scatter_add

class GraphNavierStokesLoss(torch.nn.Module):
    def __init__(self):
        super(GraphNavierStokesLoss, self).__init__()

    def forward(self, data):
        u, v, p, nu = data.x[:, 0], data.x[:, 1], data.x[:, 4], data.x[:, 5]
        edge_weights = data.edge_attr
        epsilon = 1e-8

        laplacian_indices, laplacian_values = get_laplacian(data.edge_index, edge_weight=edge_weights, normalization=None)
        num_nodes = data.num_nodes
        laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))

        du_dx = (u[data.edge_index[1]] - u[data.edge_index[0]]) / (edge_weights+epsilon)
        du_dy = (u[data.edge_index[1]] - u[data.edge_index[0]]) / (edge_weights+epsilon)
        dv_dx = (v[data.edge_index[1]] - v[data.edge_index[0]]) / (edge_weights+epsilon)
        dv_dy = (v[data.edge_index[1]] - v[data.edge_index[0]]) / (edge_weights+epsilon)

        div_u = du_dx + dv_dy
        continuity_loss = div_u.abs().mean()
        du_dx_node = scatter_add(du_dx, data.edge_index[0], dim=0, dim_size=num_nodes)
        du_dy_node = scatter_add(du_dy, data.edge_index[0], dim=0, dim_size=num_nodes)
        dv_dx_node = scatter_add(dv_dx, data.edge_index[0], dim=0, dim_size=num_nodes)
        dv_dy_node = scatter_add(dv_dy, data.edge_index[0], dim=0, dim_size=num_nodes)

        conv_u = u * du_dx_node + v * du_dy_node
        conv_v = u * dv_dx_node + v * dv_dy_node
        

        laplacian_u = torch.sparse.mm(laplacian, u.unsqueeze(-1)).squeeze()
        laplacian_v = torch.sparse.mm(laplacian, v.unsqueeze(-1)).squeeze()
        assert laplacian_u.size(0) == u.size(0)
        assert laplacian_v.size(0) == v.size(0)

        dp_dx = (p[data.edge_index[1]] - p[data.edge_index[0]]) / (edge_weights+epsilon)
        dp_dy = (p[data.edge_index[1]] - p[data.edge_index[0]]) /(edge_weights+epsilon)

        dp_dx_node = scatter_add(dp_dx, data.edge_index[0], dim=0, dim_size=num_nodes)
        dp_dy_node = scatter_add(dp_dy, data.edge_index[0], dim=0, dim_size=num_nodes)

        momentum_loss_u = (conv_u + dp_dx_node - nu * laplacian_u).abs().mean()
        momentum_loss_v = (conv_v + dp_dy_node - nu * laplacian_v).abs().mean()

        total_momentum_loss = momentum_loss_u + momentum_loss_v

        # Return the sum of the continuity loss and the total momentum loss
        return continuity_loss + total_momentum_loss






