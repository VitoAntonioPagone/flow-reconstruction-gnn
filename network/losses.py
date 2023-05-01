import torch
from torch.nn import Module


class MaskedMSELoss(Module):
    """ Custom masked MSE loss"""

    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predicted, target, mask):
        diff = predicted - target
        masked_diff = diff * mask
        loss_value = torch.mean(masked_diff ** 2)
        return loss_value

class NavierStokesLoss(Module):
    def __init__(self, model, rho=1.0, mu=1.0, delta=1e-3):
        super().__init__()
        self.model = model
        self.rho = rho
        self.mu = mu
        self.delta = delta

    def forward(self, pred, x, y, temp):
        u, v, w = torch.split(pred, 1, dim=1)

        du_dx, du_dy = self._get_derivatives(u, x, y, temp)
        dv_dx, dv_dy = self._get_derivatives(v, x, y, temp)
        dw_dx, dw_dy = self._get_derivatives(w, x, y, temp)

        div = du_dx + dv_dy + dw_dy

        d2u_dx2, d2u_dy2 = self._get_second_derivatives(u, x, y, temp)
        d2v_dx2, d2v_dy2 = self._get_second_derivatives(v, x, y, temp)
        d2w_dx2, d2w_dy2 = self._get_second_derivatives(w, x, y, temp)

        # Convective terms
        u_du_dx = u * du_dx
        v_du_dy = v * du_dy
        u_dv_dx = u * dv_dx
        v_dv_dy = v * dv_dy
        u_dw_dx = u * dw_dx
        v_dw_dy = v * dw_dy

        NS_x = self.rho * (u_du_dx + v_du_dy) - self.mu * (d2u_dx2 + d2u_dy2)
        NS_y = self.rho * (u_dv_dx + v_dv_dy) - self.mu * (d2v_dx2 + d2v_dy2)
        NS_z = self.rho * (u_dw_dx + v_dw_dy) - self.mu * (d2w_dx2 + d2w_dy2)

        NS_loss = torch.mean(torch.abs(NS_x)) + torch.mean(torch.abs(NS_y)) + torch.mean(torch.abs(NS_z))

        div_loss = torch.mean(torch.abs(div))

        return NS_loss + div_loss

    def _get_derivatives(self, f, x, y, temp):
        df_dx = (self.model(torch.cat((x + self.delta, y, temp), dim=-1))[:, 0] - f) / self.delta
        df_dy = (self.model(torch.cat((x, y + self.delta, temp), dim=-1))[:, 0] - f) / self.delta
        return df_dx, df_dy

    def _get_second_derivatives(self, f, x, y, temp):
        d2f_dx2 = (self.model(torch.cat((x + 2 * self.delta, y, temp), dim=-1))[:, 0] - 2 * self.model(torch.cat((x + self.delta, y, temp), dim=-1))[:, 0] + f) / (self.delta ** 2)
        d2f_dy2 = (self.model(torch.cat((x, y + 2 * self.delta, temp), dim=-1))[:, 0] - 2 * self.model(torch.cat((x, y + self.delta, temp), dim=-1))[:, 0] + f) / (self.delta ** 2)
        return d2f_dx2, d2f_dy2

    

