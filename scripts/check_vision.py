"""CPU checks of chapter 02 notebook definitions; no datasets or GUI required."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from check_examples import definitions

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = next(ROOT.glob('02 */notes.ipynb'))


def check():
    env = dict(torch=torch, nn=nn, F=F, np=np, Image=Image, Path=Path,
               DATA_DIR=ROOT/'datasets', optim=torch.optim)
    definitions(NOTEBOOK, ['f0ee91f9', 'e07c76d2', '0c848dcb', 'dfbbe487',
                          'bd28aa8d', '8f87d2cc', 'bd464964', 'f2855619',
                          '25e3d8ae', '014e95aa', '1eb102aa', 'f06d703b'], env)
    ImageCNN, UNet = env['ImageCNN'], env['UNet']
    for model, shape in [(ImageCNN(), (2, 1, 28, 28)),
                         (ImageCNN(3, 1), (2, 3, 32, 32)),
                         (UNet(f=(4, 8, 16, 32, 64)), (2, 3, 32, 32))]:
        x = torch.rand(shape)
        y = model(x)
        assert y.shape == (shape[0], 1, shape[2], shape[3])
        assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
        y.mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        try:
            model(torch.rand(2, shape[1], 31, 31))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid spatial sizes must fail early')

    # Evaluation aggregates actual samples, including a short final batch.
    samples = torch.utils.data.TensorDataset(torch.rand(3, 1, 28, 28), torch.tensor([0, 1, 2]))
    loader = torch.utils.data.DataLoader(samples, batch_size=2)
    net = env['FNN']()
    report = env['evaluate_classifier'](net, loader)
    with torch.no_grad():
        expected = F.nll_loss(net(samples.tensors[0]), samples.tensors[1]).item()
    assert abs(report['loss'] - expected) < 1e-6 and report['samples'] == 3

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:, 0] = 255
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, 4:] = 255
        Image.fromarray(image).save(temp/'image.tif')
        Image.fromarray(mask).save(temp/'mask.gif')
        dataset = env['SegmentationDataset'](([temp/'image.tif'], [temp/'mask.gif']), size=(16, 32))
        x, target = dataset[0]
        assert x.shape == (3, 32, 16) and target.shape == (1, 32, 16)
        assert set(target.unique().tolist()) == {0.0, 1.0}
        assert target[:, :, :8].sum() == 0 and target[:, :, 8:].min() == 1
        try:
            env['get_paths'](temp/'missing-drive')
        except FileNotFoundError as exc:
            assert '21_training.tif' in str(exc)
        else:
            raise AssertionError('Missing DRIVE files must be explained')
        # Check each original ID pairs with its own mask without downloading data.
        for split, ids in [('training', range(21, 41)), ('test', range(1, 21))]:
            (temp/split/'images').mkdir(parents=True)
            (temp/split/'1st_manual').mkdir()
            for i in ids:
                (temp/split/'images'/f'{i:02d}_{split}.tif').touch()
                (temp/split/'1st_manual'/f'{i:02d}_manual1.gif').touch()
        train, test = env['get_paths'](temp)
        assert len(train[0]) == len(test[1]) == 20
        assert test[0][0].name == '01_test.tif' and test[1][0].name == '01_manual1.gif'

    images = torch.rand(3, 3, 32, 32)
    masks = (torch.rand(3, 1, 32, 32) > 0.5).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, masks), batch_size=2)
    model = UNet(f=(4, 8, 16, 32, 64))
    before = model.outputs.weight.detach().clone()
    env['train_segmenter'](model, {'train': loader}, num_epochs=1)
    assert not torch.equal(before, model.outputs.weight)
    running_mean = model.encoder1.conv.conv[1].running_mean.clone()
    score = env['evaluate_segmentation'](model, loader)
    assert 0 <= score <= 1 and not model.training
    torch.testing.assert_close(running_mean, model.encoder1.conv.conv[1].running_mean)
    with torch.no_grad():
        predictions = model(images) >= 0.5
        truth = masks.bool()
        expected = 2 * (predictions & truth).sum() / (predictions.sum() + truth.sum())
    assert abs(score - expected.item()) < 1e-6
    empty = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images[:0], masks[:0]))
    try:
        env['evaluate_segmentation'](model, empty)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty evaluation must not report perfect Dice')

    clean = torch.full((4, 1, 28, 28), 0.5)
    base = torch.utils.data.TensorDataset(clean, torch.zeros(4, dtype=torch.long))
    noisy = env['NoisyImages'](base, seed=9)
    first, reference = noisy[0]
    torch.testing.assert_close(first, noisy[0][0])
    torch.testing.assert_close(reference, clean[0])
    assert not torch.equal(first, noisy[1][0]) and first.dtype == torch.float32
    assert first.min() >= 0 and first.max() <= 1
    torch.testing.assert_close(env['NoisyImages'](base, sigma=0)[0][0], clean[0])
    smooth = torch.ones(2, 1, 8, 8, requires_grad=True)
    loss = env['denoising_loss']()(smooth, smooth.detach(), 10)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(smooth.grad).all()
    model = ImageCNN()
    before = model.conv1[0].weight.detach().clone()
    env['train_denoiser'](model, {'train': torch.utils.data.DataLoader(noisy, batch_size=2)})
    assert not torch.equal(before, model.conv1[0].weight)

    # Real reader control flow with a tiny capture double: bounded and always released.
    class Capture:
        def __init__(self):
            self.reads, self.released = 0, False
        def isOpened(self):
            return True
        def read(self):
            self.reads += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        def release(self):
            self.released = True
    capture = Capture()
    env['cv2'] = SimpleNamespace(VideoCapture=lambda _: capture)
    assert len(env['read_video']('unused', max_frames=3)) == 3
    assert capture.reads == 3 and capture.released
    capture = Capture()
    capture.isOpened = lambda: False
    try:
        env['read_video']('missing')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Unreadable video must fail')
    assert capture.released
    print('PASS: chapter 02 shapes/gradients, sample-weighted evaluation, masks, DRIVE paths, eval mode, lazy noise, TV loss, bounded video')


if __name__ == '__main__':
    check()
