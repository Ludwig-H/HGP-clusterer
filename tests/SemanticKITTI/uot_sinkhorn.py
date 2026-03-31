import torch
from typing import Optional

def solve_uot_sinkhorn_gpu(
    C: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor, 
    epsilon: float, 
    tau1: float, 
    tau2: float, 
    iters: int = 15
) -> torch.Tensor:
    """ 
    Résout le Transport Optimal Déséquilibré régularisé par KL.
    C, a, b doivent être des torch.Tensor sur le même device (ex: 'cuda').
    """
    # K gère l'infini nativement pour le Gating : torch.exp(-inf) = 0.0
    K = torch.exp(-C / epsilon) 
    
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    exp_u = tau1 / (tau1 + epsilon)
    exp_v = tau2 / (tau2 + epsilon)
    
    for _ in range(iters):
        # Mise à jour des potentiels avec stabilisation
        u = (a / (torch.matmul(K, v) + 1e-12)) ** exp_u
        v = (b / (torch.matmul(K.T, u) + 1e-12)) ** exp_v
        
    # OPTIMISATION CRITIQUE : Utiliser le broadcasting, SURTOUT PAS torch.diag()
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P

def uot_cost_kl_gpu(
    P: torch.Tensor, 
    C: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor, 
    tau1: float, 
    tau2: float
) -> float:
    """ Calcule le coût global UOT (Distance de Wasserstein Déséquilibrée) """
    def kl_div(x, y):
        mask = (x > 1e-12) & (y > 1e-12)
        res = torch.zeros_like(x)
        res[mask] = x[mask] * torch.log(x[mask] / y[mask]) - x[mask] + y[mask]
        res[~mask] = y[~mask] # Limite mathématique quand x -> 0
        return torch.sum(res)

    p_row = torch.sum(P, dim=1)
    p_col = torch.sum(P, dim=0)
    
    # Masque pour éviter les NaN si C contient inf et P est 0
    valid_mask = P > 1e-12
    transport_cost = torch.sum(P[valid_mask] * C[valid_mask])
    
    penalty1 = tau1 * kl_div(p_row, a)
    penalty2 = tau2 * kl_div(p_col, b)
    
    return float((transport_cost + penalty1 + penalty2).item())
