"""CPU regressions for actual practice code; no downloads, models or APIs required."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[1]
CHAPTER=next(ROOT.glob('03 *'))
V=CHAPTER/'practices/vit-normalization'
M=CHAPTER/'practices/mllm-compression-safety'
sys.path.insert(0,str(V))
from train import make_model, run_epoch
from utils.scheduler import WarmupCosineSchedule


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    for variant in ('baseline','PostPre','PreB2TPost','PrePost'):
        model=make_model(variant,0.5,image_size=32,patch_size=8,num_classes=3,
                         dim=32,depth=2,heads=2,dim_head=16,mlp_dim=64)
        x=torch.randn(3,3,32,32);y=torch.tensor([0,1,2])
        batches=[(x[:2],y[:2]),(x[2:],y[2:])]
        before=model.mlp_head.weight.detach().clone()
        optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
        run_epoch(model,batches,'cpu',optimizer)
        assert model.training and not torch.equal(before,model.mlp_head.weight)
        result=run_epoch(model,batches,'cpu')
        assert not model.training
        with torch.no_grad():
            logits=model(x)
            torch.testing.assert_close(torch.tensor(result['loss']),torch.nn.functional.cross_entropy(logits,y))
            assert abs(result['accuracy']-(logits.argmax(-1)==y).float().mean().item())<1e-6
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'checkpoint.pt'
            torch.save({'model_state':model.state_dict()},checkpoint)
            model.load_state_dict(torch.load(checkpoint,weights_only=True)['model_state'])
    schedule=WarmupCosineSchedule(optimizer,2,10)
    assert schedule.lr_lambda(0)==0 and schedule.lr_lambda(2)==1 and schedule.lr_lambda(10)==0
    assert schedule.lr_lambda(20)==0
    summarize=load('practice_metrics',M/'metrics.py').summarize
    result=summarize({'0':{'ans':{'a':{'is_safe(gpt)':' SAFE '}}},
                      '1':{'ans':{'a':{'is_safe(gpt)':'unsafe'}}},'2':{}})['a']
    assert result['safe']==result['unsafe']==result['unjudged']==1
    assert result['coverage']==2/3 and result['unsafe_rate']==0.5
    assert summarize({'0':{'ans':{'a':{'is_safe(gpt)':'maybe'}}}})['a']['unsafe_rate'] is None
    assert summarize({})=={}
    guard=load('practice_run',M/'run.py').new_output
    with tempfile.TemporaryDirectory() as temp:
        try:guard(Path(temp))
        except FileExistsError:pass
        else:raise AssertionError('Existing output must not be overwritten')
    patched=load('practice_patch',M/'patch_autoawq.py').patched
    try:patched('unrecognized source')
    except ValueError:pass
    else:raise AssertionError('Unknown installed code must not be patched')
    print('PASS: 4 ViT variants, sample-weighted validation, checkpoint roundtrip, schedule, evaluation coverage, output guards')


if __name__=='__main__':check()
