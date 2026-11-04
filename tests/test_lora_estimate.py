import copy

import pytest

from solvi.experimental.lora import _warm_update_estimate


def test_cold_outlier_does_not_trigger_false_nearby_budget_refusal():
    ticks=iter([0,5,5,8,8,11,11,14]);state={'parameter':2,'gradient':None,'backwards':0}
    class Loss:
        def backward(self):state['gradient']=4;state['backwards']+=1
    def clear():state['gradient']=None
    per,diagnostic=_warm_update_estimate(Loss,clear,clock=lambda:next(ticks))
    assert per==3 and diagnostic['cold_seconds']==5 and diagnostic['warm_seconds']==[3,3,3]
    assert 5*300>1000 and per*300<1000
    assert state=={'parameter':2,'gradient':None,'backwards':4}


def test_estimation_clears_gradients_on_failure_before_optimizer_exists():
    state={'gradient':None,'cleared':0}
    class Loss:
        def backward(self):state['gradient']=1;raise RuntimeError('probe failed')
    def clear():state['gradient']=None;state['cleared']+=1
    with pytest.raises(RuntimeError,match='probe failed'):_warm_update_estimate(Loss,clear)
    assert state=={'gradient':None,'cleared':1}


def test_cpu_probe_preserves_parameters_and_exact_training_updates(monkeypatch):
    torch=pytest.importorskip('torch')
    torch.manual_seed(9)
    net=torch.nn.Linear(1,1,bias=False).eval();control=copy.deepcopy(net)
    def loss():return (net(torch.tensor([[2.]]))-1).square().mean()
    before=copy.deepcopy(net.state_dict())
    def no_cuda(*args,**kw):raise AssertionError('CPU estimate must not use CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',no_cuda)
    monkeypatch.setattr(torch.cuda,'synchronize',no_cuda)
    _warm_update_estimate(loss,lambda:net.zero_grad(set_to_none=True))
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in net.parameters())
    monkeypatch.undo()  # optimizer internals may inspect accelerator availability; the probe did not
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.01) for m in [net,control]]
    for m,opt in zip([net,control],optimizers):
        for _ in range(3):
            ((m(torch.tensor([[2.]]))-1).square().mean()).backward();opt.step();opt.zero_grad(set_to_none=True)
    assert all(torch.equal(a,b) for a,b in zip(net.parameters(),control.parameters()))
