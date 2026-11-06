import math
import torch

def masked_mae(pred,target):
    mask=torch.isfinite(target)&(target>0)
    if not mask.any(): raise ValueError("No valid positive targets")
    if not torch.isfinite(pred[mask]).all(): raise FloatingPointError("Nonfinite prediction")
    return (pred[mask]-target[mask]).abs().mean()

class MetricAccumulator:
    def __init__(self,horizon,mape_threshold=0.0):
        self.sums=torch.zeros(horizon,5,dtype=torch.float64);self.threshold=mape_threshold
    def update(self,pred,target):
        pred=pred.detach().double();target=target.detach().double()
        if pred.shape!=target.shape:raise ValueError(f"Prediction shape {pred.shape} differs from {target.shape}")
        valid=torch.isfinite(target)&(target>0)
        if not torch.isfinite(pred[valid]).all():raise FloatingPointError("Nonfinite prediction at valid targets")
        error=torch.where(valid,pred-target,0)
        pmask=valid&(target>self.threshold)
        ape=torch.where(pmask,error.abs()/torch.where(pmask,target,1),0)*100
        axes=(0,2,3)
        values=torch.stack([error.abs().sum(axes),error.square().sum(axes),ape.sum(axes),valid.sum(axes),pmask.sum(axes)],-1)
        self.sums+=values.cpu()
    @staticmethod
    def compute(s):
        ae,se,ape,n,npct=s.tolist()
        return {"MAE":ae/n if n else None,"RMSE":math.sqrt(se/n) if n else None,"MAPE":ape/npct if npct else None,"valid_count":int(n),"mape_count":int(npct)}
    def result(self):
        return {"overall":self.compute(self.sums.sum(0)),"per_horizon":{str(i+1):self.compute(s) for i,s in enumerate(self.sums)}}
