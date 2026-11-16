import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from models import EMFoundation
from training.finetune_multitask import parameter_loss

CFG={"backbone":"transformer","d_model":32,"n_layers":1,"nhead":4,"patch_size":8,
     "mae_decoder_layers":1,"head_hidden":32,"use_frequency":True,"mask_ratio":0.75}


def test_masked_mae_and_fusion():
    torch.manual_seed(5)
    m=EMFoundation(CFG)
    x=torch.randn(2,64,2);sr=torch.ones(2)*1e6
    y,target=m.forward_mae(x,sr)
    assert y.shape==target.shape==(2,6,16)
    assert torch.isfinite(y).all()
    for task,classes in (("protocol",8),("modulation",23),("parameter",7)):
        assert m(x,sr,task).shape==(2,classes)
    assert m.shared_ratio()>.8


def test_masked_loss_excludes_invalid_entries():
    p=torch.tensor([[0.,1.]])
    y=torch.tensor([[0.,999.]])
    v=torch.tensor([[1.,0.]])
    assert parameter_loss(p,y,v).item()==0
