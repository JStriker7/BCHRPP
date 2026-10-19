
import torch
import torch.nn as nn

class EvidenceEncoder(nn.Module):
    """Stage I:
    Compress modality sequences into causal evidence tokens.
    """
    def __init__(self,d=256,q=4):
        super().__init__()
        self.query=nn.Parameter(torch.randn(q,d))

    def forward(self,x,mask=None):
        score=torch.matmul(self.query,x.transpose(1,2))
        if mask is not None:
            score=score.masked_fill(~mask[:,None,:].bool(), torch.finfo(score.dtype).min)
        att=torch.softmax(score,dim=-1)
        return torch.matmul(att,x)


class CausalDisentangler(nn.Module):
    """Stage II:
    E -> causal, appearance, nuisance factors.
    """
    def __init__(self,d):
        super().__init__()
        self.c=nn.Linear(d,d)
        self.a=nn.Linear(d,d)
        self.n=nn.Linear(d,d)

    def forward(self,x):
        return self.c(x),self.a(x),self.n(x)


class CounterfactualContribution(nn.Module):
    """Stage III:
    Estimate evidence contribution by counterfactual masking."""
    def forward(self,full,masked):
        return torch.norm(full-masked,dim=-1)


class CognitiveCompetition(nn.Module):
    """Adaptive evidence resource allocation."""
    def forward(self,energy):
        return torch.softmax(energy,-1)


class CausalHypergraphTransformer(nn.Module):
    """Stage IV:
    Dynamic high-order reasoning backbone."""
    def __init__(self,d):
        super().__init__()
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d,4,batch_first=True),
            2)

    def forward(self,x):
        return self.encoder(x)


class DualArbitration(nn.Module):
    """Stage V:
    Fast perception + slow reasoning arbitration."""
    def __init__(self,d,c):
        super().__init__()
        self.fast=nn.Linear(d,c)
        self.slow=nn.Linear(d,c)
        self.gate=nn.Linear(d,1)

    def forward(self,x):
        h=x.mean(1)
        g=torch.sigmoid(self.gate(h))
        return (1-g)*self.fast(h)+g*self.slow(h)


class BCHRPP(nn.Module):
    def __init__(self,d=256,classes=7):
        super().__init__()
        self.e=EvidenceEncoder(d)
        self.dis=CausalDisentangler(d)
        self.cf=CounterfactualContribution()
        self.comp=CognitiveCompetition()
        self.hg=CausalHypergraphTransformer(d)
        self.head=DualArbitration(d,classes)

    def forward(self,t,a,v,text_mask=None,audio_mask=None,vision_mask=None):
        xs=[]
        for x,mask in [(t,text_mask),(a,audio_mask),(v,vision_mask)]:
            e=self.e(x,mask)
            c,_,_=self.dis(e)
            xs.append(c)
        z=torch.cat(xs,1)
        z=self.hg(z)
        return self.head(z)
