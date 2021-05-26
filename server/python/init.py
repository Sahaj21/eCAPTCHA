import torch.nn as nn
import torch
import torch as T
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict(data,model):
    y_hat=model.forward(data)
    pred=T.max(y_hat,axis=-1)
    return pred

device = get_default_device()


def idx_to_class(d):
    p={}
    for k in d.keys():
        p[d[k]]=k

    return p