"""Small CPU regressions using the definitions from the real notebook cells."""
from pathlib import Path
from math import sqrt
import ast
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.manual_seed(7)


def definitions(path, identifiers, namespace):
    cells = json.loads(path.read_text(encoding='utf-8'))['cells']
    cells = {cell['id']: cell for cell in cells}
    for i in identifiers:
        tree = ast.parse(''.join(cells[i]['source']))
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, f'{path.name}:{i}', 'exec'), namespace)


def check():
    # No duplicate implementation: tests extract and execute the actual teaching code.
    env = dict(torch=torch, nn=nn, F=F, sqrt=sqrt, copy=copy, device='cpu')
    path = next(ROOT.glob('03*/*.ipynb'))
    sequence_cells = {c['id']: c for c in json.loads(path.read_text(encoding='utf-8'))['cells']}
    sequence_env = {}
    for setup, forward, layers in [('70cdcc57', '6e53376d', 1), ('51084d5b', '2dc6bfb6', 2)]:
        exec(''.join(sequence_cells[setup]['source']), sequence_env)
        exec(''.join(sequence_cells[forward]['source']), sequence_env)
        assert sequence_env['output'].shape == (7, 1, 6)
        assert sequence_env['s_n'].shape == (layers, 1, 6)
    assert sequence_env['c_n'].shape == (2, 1, 6)
    definitions(path, ['legacy-03-003', 'legacy-03-004', 'legacy-03-007', 'legacy-03-008', 'legacy-03-010', 'legacy-03-011', 'legacy-03-012', 'legacy-03-013', 'legacy-03-014'], env)
    x = torch.randn(2, 5, 8)
    norm = env['LayerNorm'](8)
    torch.testing.assert_close(norm(x), F.layer_norm(x, (8,), norm.a2, norm.b2, norm.eps))
    assert torch.isfinite(env['LayerNorm'](1)(torch.ones(2, 3, 1))).all()
    attention = env['MultiHeadAttention'](8, 2)
    assert attention.W_q.out_features == 8  # All heads need independent projections.
    q, k, v = torch.randn(2, 3, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    reference = nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([attention.W_q.weight, attention.W_k.weight, attention.W_v.weight]))
        reference.in_proj_bias.copy_(torch.cat([attention.W_q.bias, attention.W_k.bias, attention.W_v.bias]))
        reference.out_proj.load_state_dict(attention.W_B.state_dict())
    torch.testing.assert_close(attention(q, k, v), reference(q, k, v, need_weights=False)[0])
    mask = torch.triu(torch.ones(1, 1, 5, 5), diagonal=1)
    before = attention(x, x, x, mask)
    changed = x.clone(); changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 10
    torch.testing.assert_close(before[:, :3], attention(changed, changed, changed, mask)[:, :3])
    projected = attention.W_q(x).reshape(2, 5, 2, 4)
    assert not torch.allclose(projected[:, :, 0], projected[:, :, 1])
    emb = env['Embeddings'](10, 8)
    assert emb(torch.tensor([[1, 2]])).shape == (1, 2, 8)
    try:
        env['PositionalEncoding'](3)
    except ValueError:
        pass
    else:
        raise AssertionError('Odd embedding dimension must be rejected')
    connection = env['SublayerConnection'](8)
    encoder = env['EncoderLayer'](1, env['EncoderBlock'](env['MultiHeadAttention'](8, 2), env['FeedForward'](8, 8, 12, 0), connection))
    decoder = env['DecoderLayer'](1, env['DecoderBlock'](env['MultiHeadAttention'](8, 2), env['MultiHeadAttention'](8, 2), env['FeedForward'](8, 8, 12, 0), connection))
    model = env['EncoderDecoder'](encoder, decoder, emb, env['Embeddings'](10, 8), env['PositionalEncoding'](8), env['Generator'](8, 10, 12, 0))
    source, target = torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2]])
    causal = torch.triu(torch.ones(1, 1, 2, 2), diagonal=1)
    out = model(source, target, causal, None)
    assert out.shape == (1, 2, 10)
    torch.testing.assert_close(out.sum(-1), torch.ones(1, 2))
    torch.testing.assert_close(out[:, :1], model(source, target[:, :1], causal[:, :, :1, :1], None))
    (-out[0, 0, 2].log()).backward()
    assert model.source_embed.embedding_layer.weight.grad is not None
    # Check that the shipped integration cells do not reintroduce causal source masks.
    demo = ''.join(next(c for c in json.loads(path.read_text(encoding='utf-8'))['cells'] if c['id'] == 'legacy-03-015')['source'])
    assignments = [node for node in ast.walk(ast.parse(demo)) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'mask2' for t in node.targets)]
    assert len(assignments) == 3 and all(isinstance(a.value, ast.Constant) and a.value.value is None for a in assignments)
    env.update(optim=torch.optim)
    definitions(next(ROOT.glob('02*/*.ipynb')), ['9dbbe5da', 'e07c76d2', '0c848dcb'], env)
    samples = torch.utils.data.TensorDataset(torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
    loaders = {key: torch.utils.data.DataLoader(samples, batch_size=2) for key in ('train', 'test')}
    env['loaders'] = loaders
    net = env['Net'](); before = net.hidden1.weight.detach().clone()
    env['train'](net, loaders, 1, 0.001, device='cpu')
    assert not torch.equal(before, net.hidden1.weight)
    env['test'](net, device='cpu')
    print('PASS: relocated RNN/LSTM examples, LayerNorm, independent attention heads, reference attention, causality, encoder/decoder backward, MNIST training')


if __name__ == '__main__':
    check()
