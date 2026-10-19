
import torch
import torch.nn.functional as F

def reconstruction_loss(x,xhat):
    return F.mse_loss(x,xhat)

def factor_orthogonal_loss(a,b):
    return torch.norm(torch.matmul(a.transpose(1,2),b))

def counterfactual_loss(p,p_cf):
    m=(p+p_cf)/2
    return 0.5*(F.kl_div(p.log(),m)+F.kl_div(p_cf.log(),m))

def total_loss(task,factor,cf,hyper,
               l1=1,l2=1,l3=1):
    return task+l1*factor+l2*cf+l3*hyper
