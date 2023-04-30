import torch.nn as nn
import torch
from torchdiffeq import odeint

class NODE(nn.Module):
    def __init__(self,hidden_dim, n_layers = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        ode_stack = [nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU()) for _ in range(n_layers)]
        self.node = nn.Sequential(*ode_stack,nn.Linear(hidden_dim,hidden_dim))
    def ode_fun(self,t,h):
        return self.node(h)

    def forward(self,end_time,start_time, h, eval_mode = False):
        if eval_mode:
            """
            When in eval mode, the NODE returns the results in between the observations
            """
            eval_times = torch.linspace(start_time,end_time, steps = 10).to(h.device)
        else:
            eval_times = torch.Tensor([start_time,end_time]).to(h.device)
        h_out = odeint(self.ode_fun,h, eval_times) 
        return h_out, eval_times